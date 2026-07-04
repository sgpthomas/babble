//! `extract::partial` implements a non-ILP-based extractor based on partial
//! orderings of learned library sets.
use crate::{
  analysis::{self, SimpleAnalysis},
  ast_node::{Arity, AstNode},
  bb_query::BBQuery,
  learn::LibId,
  runner::OperationInfo,
  schedule::{rec_cost, Schedulable},
  teachable::{BindingExpr, Teachable},
};
use bitvec::{prelude::*, vec};
use core::panic;
use egg::{
  Analysis, AstSize, CostFunction, DidMerge, EGraph, Extractor, Id, Language,
  LpCostFunction, RecExpr, Runner,
};
use lexpr::print;
use log::debug;
use nom::Or;
use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rustc_hash::FxHashMap;
use std::{
  cmp::Ordering,
  collections::{hash_map::Entry, BTreeMap, HashMap, HashSet},
  fmt::{Debug, Display},
  hash::{DefaultHasher, Hash, Hasher},
};

#[derive(Debug, Clone, Copy)]
pub struct ParetoCost {
  pub cycles: OrderedFloat<f64>,
  pub area: usize,
}

impl Default for ParetoCost {
  fn default() -> Self {
    ParetoCost {
      cycles: OrderedFloat::from(0.0),
      area: 0,
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeSet<T: Debug + Default + Clone + PartialEq + Ord + Hash> {
  pub set: HashSet<T>,
}

impl<T> ClassMatch for TypeSet<T>
where
  T: Debug + Default + Clone + PartialEq + Ord + Hash + Display,
{
  fn get_type(&self) -> Vec<String> {
    self.set.iter().map(|t| t.to_string()).collect()
  }
}

impl<T> PartialEq<T> for TypeSet<T>
where
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
{
  fn eq(&self, other: &T) -> bool {
    self.set.contains(other)
  }
}

/// A `CostSet` is a set of pairs; each pair contains a set of library
/// functions paired with the cost of the current expression/eclass
/// without the lib fns, and the cost of the lib fns themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostSet {
  /// The set of library selections and their associated costs.
  /// Invariant: sorted in ascending order of `expr_cost`, except during
  /// pruning, when it's sorted in order of `full_cost`.
  pub set: Vec<LibSel>,
}

impl PartialOrd for CostSet {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    // Compare based on the first element's cycles
    self.set.partial_cmp(&other.set)
  }
}
impl Ord for CostSet {
  fn cmp(&self, other: &Self) -> Ordering {
    // Compare based on the first element's cycles
    self.set.cmp(&other.set)
  }
}

impl CostSet {
  /// Creates a `CostSet` corresponding to introducing a nullary operation.
  #[must_use]
  pub fn new() -> CostSet {
    let mut set = Vec::with_capacity(10);
    set.push(LibSel::new());
    CostSet { set }
  }

  pub fn inc_cycles(&mut self, amount: f64) {
    for ls in &mut self.set {
      ls.inc_cycles(amount);
    }
  }

  pub fn split_cycles(&mut self, vec_len: usize) {
    for ls in &mut self.set {
      ls.split_cycles(vec_len);
    }
  }

  /// Crosses over two `CostSet`s.
  /// This is essentially a Cartesian product between two `CostSet`s (e.g. if
  /// each `CostSet` corresponds to an argument of a node) such that paired
  /// `LibSel`s have their libraries combined and costs added.
  #[must_use]
  pub fn cross(&self, other: &CostSet, lps: usize) -> CostSet {
    let mut set = Vec::new();
    for ls1 in &self.set {
      for ls2 in &other.set {
        match ls1.combine(ls2, lps) {
          None => continue,
          Some(ls) => {
            // println!("ls1: {:?}", ls1);
            // println!("ls2: {:?}", ls2);
            // println!("ls: {:?}", ls);
            if let Err(pos) = set.binary_search(&ls) {
              set.insert(pos, ls.clone());
            }
          }
        }
      }
    }

    // println!("set: {:?}", set);
    CostSet { set }
  }

  /// Combines two `CostSets` by unioning them together.
  /// Used for e.g. different `ENodes` of an `EClass`.

  pub fn combine(&mut self, other: CostSet) {
    // 使用HashSet去重，假设Cost实现了Eq和Hash
    let mut seen = HashSet::new();

    // 先合并两个CostSets
    self.set.append(&mut other.set.clone());

    // 使用HashSet去重
    self.set.retain(|cost| seen.insert(cost.clone()));

    // 排序
    self.set.sort_by(|a, b| {
      a.cycles.cmp(&b.cycles).then(a.area.cmp(&b.area).then({
        let mut a_libids = a.libs.keys().collect::<Vec<_>>();
        a_libids.sort();
        let mut b_libids = b.libs.keys().collect::<Vec<_>>();
        b_libids.sort();
        a_libids.cmp(&b_libids)
      }))
    });
  }

  /// Performs trivial partial order reduction: Only keeps the Pareto frontier
  pub fn unify(&mut self) {
    // 使用HashMap来去重，并选择最小的cycles
    let mut unique_set: HashMap<Vec<LibId>, LibSel> = HashMap::new();
    let mut new_set = self.set.clone();
    new_set.sort_by(|a, b| {
      a.cycles.cmp(&b.cycles).then(a.area.cmp(&b.area)).then({
        let mut a_libids = a.libs.keys().cloned().collect::<Vec<_>>();
        a_libids.sort();
        let mut b_libids = b.libs.keys().cloned().collect::<Vec<_>>();
        b_libids.sort();
        a_libids.cmp(&b_libids)
      })
    });
    for cand in new_set.into_iter() {
      // 将 HashMap<LibId, LibInfo> 转换为 Vec<(LibId, LibInfo)>
      let mut libs_as_vec: Vec<LibId> = cand.libs.keys().cloned().collect();
      // 排序
      libs_as_vec.sort();

      // 如果libs已存在，选择cycles最小的
      unique_set
        .entry(libs_as_vec)
        .and_modify(|existing| {
          if existing.cycles > cand.cycles {
            *existing = cand.clone();
          } else if existing.area > cand.area {
            *existing = cand.clone();
          }
        })
        .or_insert(cand);
    }

    // 最后把去重后的集合赋值回self.set
    // let mut unique_set: Vec<LibSel> = Vec::new();
    self.set = unique_set.into_iter().map(|(_, v)| v).collect();
    // // 按照cycles和area排序
    self.set.sort_by(|a, b| {
      a.cycles.cmp(&b.cycles).then(a.area.cmp(&b.area)).then({
        let mut a_libids = a.libs.keys().cloned().collect::<Vec<_>>();
        a_libids.sort();
        let mut b_libids = b.libs.keys().cloned().collect::<Vec<_>>();
        b_libids.sort();
        a_libids.cmp(&b_libids)
      })
    });
  }

  #[must_use]
  pub fn add_lib(
    &self,
    lib: LibId,
    latency_acc: f64,
    latency_cpu: f64,
    area: usize,
    search_result: usize,
    exe_count: f64,
    id: Id,
    // 没有嵌套lib，不再使用
    // nested_libs: &CostSet,
    lps: usize,
  ) -> CostSet {
    // To add a lib, we do a modified cross.
    let mut set = Vec::new();

    for ls2 in &self.set {
      match ls2.add_lib(
        lib,
        latency_acc,
        latency_cpu,
        area,
        search_result,
        exe_count,
        id,
        lps,
      ) {
        None => continue,
        Some(ls) => {
          if let Err(pos) = set.binary_search(&ls) {
            set.insert(pos, ls);
          }
        }
      }
    }

    CostSet { set }
  }

  pub fn prune(&mut self, beam_size: usize) {
    if beam_size == 0 {
      self.set.clear();
      return;
    }

    // 1. 先从所有解里筛出非支配解集合 Pareto
    let mut pareto: Vec<LibSel> = Vec::new();
    for cand in &self.set {
      if !self.set.iter().any(|other| {
        // other 支配 cand？
        (other.cycles <= cand.cycles && other.area <= cand.area)
          && (other.cycles < cand.cycles || other.area < cand.area)
      }) {
        pareto.push(cand.clone());
      }
    }

    // 2. 如果非支配解多于 beam_size，按 cycles+area 排序截断
    if pareto.len() > beam_size {
      pareto.sort_by(|a, b| {
        a.cycles.cmp(&b.cycles).then(a.area.cmp(&b.area))
        // （可加 libs id 作为 tiebreaker）
      });
      pareto.truncate(beam_size);
      self.set = pareto;
      return;
    }

    // 3. 如果不够 beam_size，就在剩余解里按 cycles+area 排序补充
    let mut rest: Vec<LibSel> = self
      .set
      .drain(..)
      .filter(|cand| !pareto.iter().any(|p| p.libs == cand.libs))
      .collect();
    rest.sort_by(|a, b| a.cycles.cmp(&b.cycles).then(a.area.cmp(&b.area)));

    // 合并，并截断到 beam_size
    pareto.extend(rest.into_iter());
    pareto.truncate(beam_size);
    self.set = pareto;
  }

  // pub fn update_cost(&mut self, exe_count: usize) {
  //   for ls in &mut self.set {
  //     for (_, info) in &mut ls.libs {
  //       let latency_acc = info.latency_acc;
  //       let set = &mut info.instances;
  //       for (_id, count) in set.iter_mut() {
  //         if *count == 0 {
  //           ls.cycles +=
  //             latency_acc * OrderedFloat::from((exe_count - *count) as f64);
  //           *count = exe_count;
  //         }
  //       }
  //     }
  //   }
  // }

  pub fn unify2(&mut self) {
    // println!("unify");
    let mut i = 0;

    while i < self.set.len() {
      let mut j = i + 1;

      while j < self.set.len() {
        let ls1 = &self.set[i];
        let ls2 = &self.set[j];

        if ls2.is_subset(ls1) {
          self.set.remove(j);
        } else {
          j += 1;
        }
      }
      i += 1;
    }
  }
}

/// A'LibInfo' is a tuple containing the latency, area, and a set of instances
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibInfo {
  pub latency_acc: OrderedFloat<f64>,
  pub latency_cpu: OrderedFloat<f64>,
  pub area: usize,
  /// The instances of this library function in the expression.
  /// The key is the Id of the instance, and the value is the number of times
  /// it appears.
  pub instances: HashMap<Id, OrderedFloat<f64>>,
}

// 为LibInfo实现Hash
impl Hash for LibInfo {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.latency_acc.hash(state);
    self.latency_cpu.hash(state);
    self.area.hash(state);
    // 对instances的keys进行哈希
    let mut keys: Vec<&Id> = self.instances.keys().collect();
    keys.sort(); // Sort to ensure consistent ordering
    for key in keys {
      key.hash(state);
    }
  }
}

/// A `LibSel` is a selection of library functions, paired with two
/// corresponding cost values: the cost of the expression without the library
/// functions, and the cost of the library functions themselves
#[derive(Debug, Clone)]
pub struct LibSel {
  pub cycles: OrderedFloat<f64>,
  pub area: usize,
  /// The libraries used in this expression. Each library is binded with its
  /// latency, area and the instances.
  pub libs: HashMap<LibId, LibInfo>,
}

impl Eq for LibSel {}

impl PartialEq for LibSel {
  fn eq(&self, other: &Self) -> bool {
    self.cycles == other.cycles
      && self.area == other.area
      && self.libs == other.libs
  }
}

impl Hash for LibSel {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.cycles.hash(state);
    self.area.hash(state);

    // 只需要对libs的keys进行哈希
    let mut keys: Vec<&LibId> = self.libs.keys().collect();
    keys.sort(); // Sort to ensure consistent ordering
    for key in keys {
      key.hash(state);
    }
  }
}

impl PartialOrd for LibSel {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    let r = self.cycles.partial_cmp(&other.cycles);
    match r {
      Some(ord) => Some(ord),
      None => None,
    }
  }
}

impl Ord for LibSel {
  fn cmp(&self, other: &Self) -> Ordering {
    let r = self.cycles.partial_cmp(&other.cycles);
    match r {
      Some(ord) => ord,
      None => {
        Ordering::Equal // If we can't compare, treat as equal
      }
    }
  }
}

impl LibSel {
  #[must_use]
  pub fn new() -> LibSel {
    LibSel {
      cycles: 0.0.into(),
      area: 0,
      libs: HashMap::new(),
    }
  }

  pub fn inc_cycles(&mut self, amount: f64) {
    self.cycles += OrderedFloat::from(amount);
  }

  pub fn split_cycles(&mut self, vec_len: usize) {
    self.cycles /= OrderedFloat::from(vec_len as f64);
  }

  /// Combines two `LibSel`s. Unions the lib sets, adds
  /// the expr
  #[must_use]
  pub fn combine(&self, other: &LibSel, lps: usize) -> Option<LibSel> {
    let mut res = self.clone();

    for (lib_id, lib_info) in &other.libs {
      match res.libs.entry(*lib_id) {
        Entry::Occupied(mut entry) => {
          let set = &mut entry.get_mut().instances;
          for (id, count) in lib_info.instances.iter() {
            if let None = set.get_mut(id) {
              set.insert(*id, *count);
            } else {
              // use the same lib
              res.cycles -= lib_info.latency_acc * (*count);
            }
          }
        }
        Entry::Vacant(_entry) => {
          continue;
        }
      }
    }
    // 更新完所有旧libs之后，再更新新的lib
    for (lib_id, lib_info) in &other.libs {
      match res.libs.entry(*lib_id) {
        Entry::Occupied(_entry) => {
          continue;
        }
        Entry::Vacant(entry) => {
          entry.insert(lib_info.clone());
          res.area += lib_info.area;

          // 修复，不能超过lps就返回None，这样如果原来就有lps个，
          // 那么就会直接返回空 这会导致无法添加新的lib
          // 我们需要更改策略，如果超过lps个(此时必然是lps+1个)，
          // 那么就需要替换掉原来中的一个
          if res.libs.len() > lps {
            let mut gain_map = Vec::new();
            // 计算每个lib的gain
            for (lib_id, lib_info) in &res.libs {
              let exe_count_sum =
                lib_info.instances.values().sum::<OrderedFloat<f64>>();
              let gain =
                (lib_info.latency_cpu - lib_info.latency_acc) * exe_count_sum;
              gain_map.push((*lib_id, gain, lib_info.area));
            }
            // 按照gain排序
            gain_map.sort_by(|a, b| {
              a.1
                .partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then(a.2.cmp(&b.2))
                .then(a.0.cmp(&b.0)) // 按照lib_id排序，确保稳定性
            });

            if let Some((min_lib_id, _, _)) = gain_map.first() {
              // 删除这个lib
              res.libs.remove(min_lib_id);
              res.area -= gain_map[0].2;
              // 加上对应的gain
              res.cycles += gain_map[0].1;
            } else {
              return None; // 如果没有找到最小的gain，返回None
            }
          }
        }
      }
    }
    res.cycles += other.cycles;

    Some(res)
  }

  #[must_use]
  pub fn add_lib(
    &self,
    lib: LibId,
    latency_acc: f64,
    latency_cpu: f64,
    area: usize,
    _search_result: usize,
    exe_count: f64,
    id: Id,
    // // 没有嵌套lib，不再使用
    // _nested_libs: &LibSel,
    lps: usize,
  ) -> Option<LibSel> {
    let mut res = self.clone();

    match res.libs.entry(lib) {
      Entry::Occupied(mut entry) => {
        let set = &mut entry.get_mut().instances;
        set.insert(id, OrderedFloat::from(exe_count));
      }
      Entry::Vacant(entry) => {
        let mut set = HashMap::new();
        set.insert(id, OrderedFloat::from(exe_count));
        let info = LibInfo {
          latency_acc: OrderedFloat::from(latency_acc),
          latency_cpu: OrderedFloat::from(latency_cpu),
          area,
          instances: set,
        };
        entry.insert(info);
        res.area += area;
        if res.libs.len() > lps {
          // 同理，不能直接返回None，替换掉原来的gain最小的是最明智的选择
          let mut gain_map = Vec::new();
          // 计算每个lib的gain
          for (lib_id, lib_info) in &res.libs {
            let gain = (lib_info.latency_cpu - lib_info.latency_acc)
              * OrderedFloat::from(lib_info.instances.len() as f64);
            gain_map.push((*lib_id, gain, lib_info.area));
          }
          // 按照gain排序
          gain_map.sort_by(|a, b| {
            a.1
              .partial_cmp(&b.1)
              .unwrap_or(Ordering::Equal)
              .then(a.2.cmp(&b.2))
              .then(a.0.cmp(&b.0)) // 按照lib_id排序，确保稳定性
          });
          // 取出最小的gain的lib
          if let Some((min_lib_id, _, _)) = gain_map.first() {
            // 删除这个lib
            res.libs.remove(min_lib_id);
            res.area -= gain_map[0].2;
            // 加上对应的gain
            res.cycles += gain_map[0].1;
          } else {
            return None; // 如果没有找到最小的gain，返回None
          }
        }
      }
    }

    res.cycles += OrderedFloat::from(latency_acc as f64);

    // Keep cycles as an estimated execution cost only. `search_result` is a
    // noisy reuse signal from e-graph matching and should not directly change
    // the Pareto objective.

    Some(res)
  }

  /// O(n) subset check
  #[must_use]
  pub fn is_subset(&self, other: &LibSel) -> bool {
    // For every element in this LibSel, we want to see
    // if it exists in other.
    for (lib_id, _) in &self.libs {
      match other.libs.get(lib_id) {
        Some(_) => {}
        None => {
          return false;
        }
      }
    }
    true
  }
}

// --------------------------------
// --- The actual Analysis part ---
// --------------------------------
use std::marker::PhantomData;
#[derive(Debug, Clone)]
pub struct ISAXAnalysis<Op, T>
where
  Op: Clone + Debug + Ord + std::hash::Hash + Teachable + Arity,
{
  /// The number of `LibSel`s to keep per `EClass`.
  beam_size: usize,
  inter_beam: usize,
  /// The maximum number of libs per lib selection. Any lib selections with a
  /// larger amount will be pruned.
  lps: usize,
  bb_query: BBQuery,
  /// a map to store the true lat_acc for (lib_id, bbs)
  lat_acc_map: HashMap<(usize, Vec<String>), f64>,
  /// a map to store the length of matches for a lib
  search_len_map: HashMap<usize, usize>,
  /// Marker to indicate that this struct uses the Op type parameter
  op_phantom: PhantomData<Op>,
  ty_phantom: PhantomData<T>,
}

impl<Op, T> ISAXAnalysis<Op, T>
where
  Op: Clone + Debug + Ord + std::hash::Hash + Teachable + Arity,
{
  #[must_use]
  pub fn new(
    beam_size: usize,
    inter_beam: usize,
    lps: usize,
    bb_query: BBQuery,
  ) -> ISAXAnalysis<Op, T> {
    ISAXAnalysis {
      beam_size,
      inter_beam,
      lps,
      bb_query,
      lat_acc_map: HashMap::new(),
      search_len_map: HashMap::new(),
      op_phantom: PhantomData,
      ty_phantom: PhantomData,
    }
  }

  #[must_use]
  pub fn empty() -> ISAXAnalysis<Op, T> {
    ISAXAnalysis {
      beam_size: 0,
      inter_beam: 0,
      lps: 1,
      bb_query: BBQuery::default(),
      lat_acc_map: HashMap::new(),
      search_len_map: HashMap::new(),
      op_phantom: PhantomData,
      ty_phantom: PhantomData,
    }
  }

  pub fn with_lat_map(
    &mut self,
    lat_map: HashMap<(usize, Vec<String>), f64>,
  ) -> &mut Self {
    self.lat_acc_map = lat_map;
    self
  }

  #[must_use]
  pub fn get_lat_acc(&self, lib_id: usize, bbs: Vec<String>) -> Option<f64> {
    // 如果lat_acc_map中有对应的key，则返回对应的值
    if let Some(lat_acc) = self.lat_acc_map.get(&(lib_id, bbs)) {
      Some(*lat_acc)
    } else {
      // 否则返回0
      None
    }
  }

  pub fn with_search_len_map(
    &mut self,
    search_len_map: HashMap<usize, usize>,
  ) -> &mut Self {
    self.search_len_map = search_len_map;
    self
  }

  #[must_use]
  pub fn get_search_len(&self, lib_id: usize) -> usize {
    // 如果search_len_map中有对应的key，则返回对应的值
    if let Some(search_len) = self.search_len_map.get(&lib_id) {
      *search_len
    } else {
      // 否则返回1
      1
    }
  }
}

impl<Op, T> Default for ISAXAnalysis<Op, T>
where
  Op: Clone + Debug + Ord + std::hash::Hash + Teachable + Arity,
{
  fn default() -> Self {
    ISAXAnalysis::empty()
  }
}

// 定义StructuralHash ,结构化哈希包含cls_hash和subtree_hash,均为u64
#[derive(Debug, Clone)]
pub struct StructuralHash {
  pub cls_hash: u64,
  pub subtree_levels: BitVec<u64, Lsb0>,
  pub level_conflict: LevelConflictState,
}

impl Default for StructuralHash {
  fn default() -> Self {
    Self {
      cls_hash: 0,
      subtree_levels: bitvec![u64, Lsb0; 0; 64], // 64位初始化为0
      level_conflict: LevelConflictState::new(),
    }
  }
}

impl StructuralHash {
  pub fn merge(&mut self, other: &Self) {
    // 首先合并cls_hash
    // 为了和enode进行区分，会使用加盐值和rotate的方法打散哈希
    let mut state = DefaultHasher::new();
    let cls_hashes = [self.cls_hash, other.cls_hash];
    for &h in &cls_hashes {
      // 加salt混合 or shift
      (h.rotate_left((h % 13) as u32)).hash(&mut state);
    }
    let merged_cls_hash = state
      .finish()
      .rotate_left((cls_hashes.len() as u32 * 7) % 64);
    // 之后合并subtree_levels
    let mut merged_subtree_levels = bitvec!(u64, Lsb0; 0; 64);
    for i in 0..64 {
      if self.subtree_levels[i] & other.subtree_levels[i] {
        self.level_conflict.update(i, true);
        merged_subtree_levels.set(i, true);
      } else if self.subtree_levels[i] ^ other.subtree_levels[i] {
        self.level_conflict.update(i, false);
        let prob = self.level_conflict.get(i);
        let mut rng = StdRng::seed_from_u64(1314114514);
        if rng.r#gen::<f32>() > prob {
          merged_subtree_levels.set(i, true);
        }
      }
    }
    self.cls_hash = merged_cls_hash;
    self.subtree_levels = merged_subtree_levels;
  }
}

#[derive(Debug, Clone)]
pub struct LevelConflictState {
  // 本结构体用于在不同的enode进行合并时使用，记录不同层级的冲突状态
  states: [f32; 64],
  // 学习率
  alpha: f32,
}
impl LevelConflictState {
  pub fn new() -> Self {
    Self {
      states: [0.5; 64], // 初始化0.5的冲突概率
      alpha: 0.1,
    }
  }

  pub fn update(&mut self, level: usize, is_real_conflict: bool) {
    self.states[level] = (1.0 - self.alpha) * self.states[level]
      + self.alpha * is_real_conflict as u8 as f32 * 0.5;
  }

  pub fn get(&self, level: usize) -> f32 {
    self.states[level]
  }
}

#[derive(Debug, Clone)]
pub struct ISAXCost<T>
where
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
{
  pub cs: CostSet,
  pub ty: T,
  pub bb: Vec<String>,
  pub vec_len: usize,
}

impl<T> PartialEq for ISAXCost<T>
where
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
{
  fn eq(&self, other: &Self) -> bool {
    // self.cs == other.cs
    self.ty == other.ty && self.bb == other.bb && self.vec_len == other.vec_len
  }
}

impl<T: PartialEq> PartialEq<T> for ISAXCost<T>
where
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
{
  fn eq(&self, other: &T) -> bool {
    self.ty == *other
  }
}

impl<T: Debug + Default + Clone + PartialEq + Ord + Hash> ISAXCost<T> {
  #[must_use]
  pub fn new(cs: CostSet, ty: T, bb: Vec<String>, vec_len: usize) -> Self {
    ISAXCost {
      cs,
      ty,
      bb,
      vec_len,
    }
  }

  #[must_use]
  pub fn empty() -> Self {
    ISAXCost {
      cs: CostSet::new(),
      ty: T::default(),
      bb: Vec::new(),
      vec_len: 0,
    }
  }
}

// 计算层级哈希映射
pub fn compute_hash_level<Op: Hash>(
  enode: &AstNode<Op>,
  child_hashes: &[u64],
) -> usize
where
  Op: Teachable,
{
  let mut hasher = DefaultHasher::new();
  let op = enode.operation().to_shielding_op();
  op.hash(&mut hasher);
  for &h in child_hashes {
    h.hash(&mut hasher);
  }
  (hasher.finish() % 64) as usize // 取模映射到固定范围
}
/// 计算不考虑层级的精确哈希
pub fn compute_full_hash<Op: Hash>(
  enode: &AstNode<Op>,
  child_hashes: &[u64],
) -> u64
where
  Op: Teachable,
{
  let mut hasher = DefaultHasher::new();
  let op = enode.operation().to_shielding_op();
  op.hash(&mut hasher);
  for &h in child_hashes {
    hasher.write_u64(h);
  }
  hasher.finish()
}

impl<Op, T> Analysis<AstNode<Op>> for ISAXAnalysis<Op, T>
where
  Op: Ord
    + std::hash::Hash
    + Debug
    + Teachable
    + Arity
    + Eq
    + Clone
    + Send
    + Sync
    + 'static
    + OperationInfo,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  type Data = ISAXCost<T>;

  fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
    // println!("merge ");
    // println!("{:?}", to);
    // println!("{:?}", &from);
    let a0 = to.clone();
    // to和from的bb信息合并，并排序
    to.bb.extend(from.bb.clone());
    to.bb.sort();
    to.bb.dedup(); // 去重

    // Merging consists of combination, followed by unification and beam
    // pruning.
    to.cs.combine(from.cs.clone());

    // let exe_count = match to.bb.len() {
    //   0 => 1,
    //   _ => match self.bb_query.get(&to.bb[0]) {
    //     Some(bb) => {
    //       // println!("bb: {:?}, exe count: {}", bb.name,bb.execution_count);
    //       bb.execution_count
    //     }
    //     None => 1,
    //   },
    // };
    // to.cs.update_cost(exe_count);

    to.cs.unify();
    // println!("unified cost set: {:?}", to.cs);
    to.cs.prune(self.beam_size);
    // println!("pruned cost set: {:?}", to.cs);
    // we also need to merge the type information
    (*to).ty = AstNode::merge_types(&to.ty, &from.ty);
    // 合并哈希
    // 合并ecls哈希，目前取最大值
    // to.hash.cls_hash = max(to.hash.cls_hash, from.hash.cls_hash);
    // to.hash.subtree_levels |= from.clone().hash.subtree_levels;
    // if to.hash.cls_hash == 9008984487884026875 {
    //   panic!(
    //     "Found a hash collision , 19, a0.cls_hash: {}, from.cls_hash: {}",
    //     a0.hash.cls_hash, from.hash.cls_hash,
    //   );
    // }
    // 合并v
    // println!("{:?}", to);
    // println!("Dismerge: {} {}", &a0 != to, to != &from);

    // TODO: be more efficient with how we do this
    // println!("merge done");
    // if to.cs.set.len() > 0 {
    //   println!("{}", to.cs.set[0].latency_gain);
    // }
    to.vec_len = from.vec_len.max(to.vec_len);
    // println!("after merge: {:?}", to);
    DidMerge(&a0 != to, to != &from)
    // DidMerge(false, false)
  }

  fn make(
    egraph: &mut EGraph<AstNode<Op>, Self>,
    enode: &AstNode<Op>,
  ) -> Self::Data {
    // calculate the type
    // println!("begin make");
    let children = enode.children().to_vec();
    // 按照operation进行排序，如果operation可交换
    // if enode.operation().is_commutative() {
    //   children.sort_by_key(|&child| egraph[child].data.hash.cls_hash);
    // }
    let child_types: Vec<T> = children
      .iter()
      .map(|&child| egraph[child].data.ty.clone())
      .collect();
    // println!("operation: {:?}", enode.operation());
    // // 子节点op
    // for child in &children {
    //   println!(
    //     "child: {:?}, type: {:?}",
    //     egraph[*child].nodes, egraph[*child].data.ty
    //   );
    // }
    let ty = enode.get_rtype(&child_types);
    // // 计算子节点哈希
    // let child_hashes = children
    //   .iter()
    //   .map(|&child| egraph[child].data.hash.cls_hash.clone())
    //   .collect::<Vec<_>>();
    // // 计算当前节点哈希
    // let current_level = compute_hash_level(enode, &child_hashes);
    // // 合并子树层级
    // let mut subtree_levels = bitvec![u64, Lsb0; 0; 64];
    // subtree_levels.set(current_level, true);
    // for child in children.iter() {
    //   let child_level = egraph[*child].data.hash.subtree_levels.clone();
    //   subtree_levels |= child_level;
    // }
    // // 计算完整哈希
    // let extract_hash = compute_full_hash(enode, &child_hashes);

    // let hash = StructuralHash {
    //   cls_hash: extract_hash,
    //   subtree_levels,
    //   level_conflict: LevelConflictState::new(),
    // };
    // Calculate `vec_len`
    let children_vec_len = children
      .iter()
      .map(|&child| egraph[child].data.vec_len)
      .max()
      .unwrap_or(1);
    let mut vec_len = enode.operation().get_vec_len();
    // 如果是vec节点，那么取vec_len的最大值，否则直接设置成1
    if enode.operation().is_vector_op() {
      vec_len = vec_len.max(children_vec_len);
    } else {
      vec_len = 1;
    }

    let x = |i: &Id| &egraph[*i].data.cs;

    let self_ref = &egraph.analysis;
    // println!("begin cal cost");

    let mut exe_count = 1;
    let mut op_latency = 1.0;

    // let is_vector_op = enode.operation().is_vector_op();
    let bbs = enode.operation().get_bbs_info();
    if bbs.len() > 0 {
      if let Some(bb_entry) = self_ref.bb_query.get(&bbs[0]) {
        exe_count = bb_entry.execution_count;
        op_latency = bb_entry.cpo;
      }
    }
    op_latency *= vec_len as f64;
    if !enode.operation().is_op() {
      op_latency = 0.0;
    }

    match Teachable::as_binding_expr(enode) {
      Some(BindingExpr::Lib(id, _, b, lat_cpu, lat_acc, area)) => {
        if lat_cpu <= lat_acc || area == 0 {
          return ISAXCost::new(x(b).clone(), ty, bbs, vec_len);
        }
        // 从analysis中取出在egraph中匹配了多少次
        let search_result = egraph.analysis.get_search_len(id.0);
        // 固连的latency是错误的，需要进一步schedule计算，
        // 首先从f中利用extractor获取完整的表达式
        let b_bbs = egraph[*b].data.bb.clone();
        let mut b_exe_count = self_ref.bb_query.get_factor();
        if b_bbs.len() > 0 {
          if let Some(bb_entry) = self_ref.bb_query.get(&b_bbs[0]) {
            b_exe_count = bb_entry.execution_count_normalized;
          }
        }
        let mut e = x(b).add_lib(
          id,
          lat_acc.0,
          lat_cpu.0, // as f64 / vec_len as f64,
          area,
          search_result,
          b_exe_count,
          *b,
          // x(f),
          self_ref.lps,
        );
        // println!("new cost set: {:#?}", e);
        // if exe_count > 0 {
        //   e.update_cost(exe_count);
        // }
        e.unify();
        e.prune(self_ref.inter_beam);
        // println!("make done");
        ISAXCost::new(e, ty, bbs, vec_len)
      }
      Some(_) | None => {
        // This is some other operation of some kind.
        // We test the arity of the function

        if enode.is_empty() {
          // println!("make done");
          // 0 args. Return new.
          let mut cs = CostSet::new();
          cs.inc_cycles(op_latency * exe_count as f64);
          // cs.split_cycles(vec_len);
          ISAXCost::new(cs, ty, enode.operation().get_bbs_info(), vec_len)
        } else if enode.args().len() == 1 {
          // 1 arg. Get child cost set, inc, and return.
          let mut e = x(&enode.args()[0]).clone();
          // if exe_count > 0 {
          //   e.update_cost(exe_count);
          // }

          e.inc_cycles(op_latency * exe_count as f64);
          // 如果是get_vec节点，那么就需要除以vec_len
          if enode.operation().is_get_from_vec() {
            // 拿到子节点的vec_len
            let child_vec_len = egraph[enode.args()[0]].data.vec_len;
            e.split_cycles(child_vec_len);
          }

          ISAXCost::new(e, ty, bbs, vec_len)
        } else {
          // 2+ args. Cross/unify time!
          // 先收集所有子节点的cost set，之后排序，防止不确定性
          let mut args: Vec<_> = enode.args().iter().map(|i| x(i)).collect();
          args.sort_by(|a, b| a.cmp(b));
          let mut e = args[0].clone();
          // println!("begin cross,args.len: {}", enode.args().len());
          // println!("{:#?}", e);
          // if args.len() == 7 {
          //   println!("begin cross, cs: {:#?}", e);
          // }
          for cs in &args[1..] {
            // println!("begin prune");
            e = e.cross(cs, self_ref.lps);
            e.unify();
            e.prune(self_ref.inter_beam);
          }

          // e.prune(self_ref.inter_beam);

          // println!("inter beam: {}", self_ref.inter_beam);
          // println!("e.len after update: {}", e.set.len());
          e.inc_cycles(op_latency * exe_count as f64);
          // // 如果是vec节点，那么就需要除以vec_len
          // if enode.operation().is_vec() {
          //   e.split_cycles(vec_len);
          // }
          // println!("make done");
          // e.split_cycles(vec_len);
          // println!("make done");
          // println!("e: {:?}", e);
          ISAXCost::new(e, ty, bbs, vec_len)
        }
      }
    }
  }
}

/// Library context is a set of library function names.
/// It is used in the extractor to represent the fact that we are extracting
/// inside (nested) library definitions.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LibContext {
  set: Vec<LibId>,
  hash: u32,
}

impl LibContext {
  fn new() -> Self {
    let mut ctx = Self {
      set: Vec::new(),
      hash: 0,
    };
    ctx.cal_hash();
    ctx
  }

  /// Add a new lib to the context if not yet present,
  /// keeping it sorted.
  fn add(&mut self, lib_id: LibId) {
    if let Err(pos) = self.set.binary_search(&lib_id) {
      self.set.insert(pos, lib_id);
      self.cal_hash();
    }
  }

  /// Does the context contain the given lib?
  fn contains(&self, lib_id: LibId) -> bool {
    self.set.binary_search(&lib_id).is_ok()
  }

  /// Calculate hash
  fn cal_hash(&mut self) {
    let mut hasher = DefaultHasher::new();
    self.set.hash(&mut hasher);
    let full_hash = hasher.finish();
    self.hash = (full_hash ^ (full_hash >> 32)) as u32;
  }
}

impl Hash for LibContext {
  fn hash<H: Hasher>(&self, state: &mut H) {
    // 如果设计正确，这里应该用预计算的hash
    // 但需要确保dirty时为false！
    state.write_u32(self.hash);
  }
}

type MaybeExpr<Op> = Option<RecExpr<AstNode<Op>>>;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CacheKey {
  id: u32,   // 假设Id是u32或类似
  hash: u32, // 如果可能，减小哈希位数
}

/// Extractor that minimizes AST size but ignores the cost of library
/// definitions (which will be later lifted to the top).
/// The main difference between this and a standard extractor is that
/// instead of finding the best expression *per eclass*,
/// we need to find the best expression *per eclass and lib context*.
/// This is because when extracting inside library definitions,
/// we are not allowed to use those libraries;
/// so the best expression is different depending on which library defs we are
/// currently inside.
#[derive(Debug)]
pub struct LibExtractor<
  'a,
  Op: Clone
    + std::fmt::Debug
    + std::hash::Hash
    + Ord
    + Teachable
    + std::fmt::Display
    + OperationInfo,
  N: Analysis<AstNode<Op>>,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  LA,
  LD,
> where
  AstNode<Op>: TypeInfo<T> + Schedulable<LA, LD>,
{
  /// Remembers the best expression so far for each pair of class id and lib
  /// context; if an entry is absent, we haven't visited this class in this
  /// context yet; if an entry is `None`, it's currently under processing,
  /// but we have no results for it yet; if an entry is `Some(_)`, we have
  /// found an expression for it (but it might still be improved).
  memo: FxHashMap<CacheKey, usize>,
  all_exprs: Vec<MaybeExpr<Op>>,
  all_costs: Vec<Option<OrderedFloat<f64>>>,
  /// Current lib context:
  /// contains all lib ids inside whose definitions we are currently
  /// extracting.
  lib_context: LibContext,
  /// The egraph to extract from.
  egraph: &'a EGraph<AstNode<Op>, N>,
  /// This is here for pretty debug messages.
  indent: usize,
  /// The relative weight of area cost and delay cost, from 0.0 (all areaa) to
  /// 1.0 (all delay)
  bb_query: BBQuery,
  _phantom: PhantomData<(T, LA, LD)>,
}

impl<'a, Op, N, T, LA, LD> LibExtractor<'a, Op, N, T, LA, LD>
where
  Op: Clone
    + std::fmt::Debug
    + std::hash::Hash
    + Ord
    + Teachable
    + std::fmt::Display
    + Arity
    + OperationInfo,
  N: Analysis<AstNode<Op>> + Clone,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T> + Schedulable<LA, LD>,
{
  /// Create a lib extractor for the given egraph
  pub fn new(egraph: &'a EGraph<AstNode<Op>, N>, bb_query: BBQuery) -> Self {
    Self {
      memo: FxHashMap::default(),
      all_exprs: Vec::with_capacity(10000),
      all_costs: Vec::with_capacity(10000),
      lib_context: LibContext::new(),
      egraph,
      indent: 0,
      bb_query,
      _phantom: PhantomData,
    }
  }

  /// Get best best expression for `id` in the current lib context.
  fn get_from_memo(&self, id: Id) -> Option<&usize> {
    // let start = Instant::now();
    let index = self.memo.get(&CacheKey {
      id: usize::from(id) as u32,
      hash: self.lib_context.hash,
    });
    // println!("get_from_memo: {}ms", start.elapsed().as_millis());
    index
  }

  /// Set best best expression for `id` in the current lib context.
  fn insert_into_memo(
    &mut self,
    id: Id,
    val: MaybeExpr<Op>,
    cost: Option<OrderedFloat<f64>>,
  ) {
    // let start = Instant::now();
    // 使用prune进行修剪
    let new_val = self.prune(val.clone());
    // println!("id: {}, cost: {:?}", id, cost);
    self.all_exprs.push(new_val);
    self.all_costs.push(cost);
    self.memo.insert(
      CacheKey {
        id: usize::from(id) as u32,
        hash: self.lib_context.hash,
      },
      self.all_exprs.len() - 1,
    );
    // println!("inserted into memo: {}ms", start.elapsed().as_millis());
  }

  /// prune using egraph
  fn prune(&self, val: MaybeExpr<Op>) -> MaybeExpr<Op> {
    // 如果val为None，直接返回
    if val.is_none() {
      return val;
    }
    // 如果val不为None,取出RecExpr
    let timeout = std::time::Duration::from_millis(60);
    let expr = val.clone().unwrap();
    let mut egraph = EGraph::new(SimpleAnalysis::default());
    egraph.add_expr(&expr);
    egraph.rebuild();
    let runner = Runner::<_, _, ()>::new(SimpleAnalysis::default())
      .with_expr(&expr)
      .with_time_limit(timeout)
      .with_iter_limit(4)
      .run(vec![]);
    // 提取出最优的表达式
    let egraph = runner.egraph;
    let extractor = Extractor::new(&egraph, AstSize);
    Some(extractor.find_best(runner.roots[0]).1)
  }

  /// Extract the smallest expression for the eclass `id`.
  /// # Panics
  /// Panics if extraction fails
  /// (this should never happen because the e-graph must contain a non-cyclic
  /// expression)
  pub fn best(&mut self, id: Id) -> RecExpr<AstNode<Op>> {
    // Populate the memo:
    //println!("extracting eclass {id}");
    self.extract(id);
    // println!("id: {:#?}", id);
    // println!("{:#?}", self.egraph[id]);
    // Get the best expression from the memo:
    let index = self.get_from_memo(id).unwrap().clone();
    self.all_exprs[index]
      .clone()
      .expect("Failed to extract expression")
  }

  /// Expression gain used by this extractor
  pub fn cost(&self, expr: &RecExpr<AstNode<Op>>) -> ParetoCost {
    let (cycles, area) = rec_cost(expr, &self.bb_query, HashMap::new());
    // println!("cycles: {}", cycles);
    ParetoCost {
      cycles: OrderedFloat::from(cycles),
      area,
    }
  }

  /// Extract the expression with the largest gain from the eclass id and its
  /// descendants in the current context, storing results in the memo
  fn extract(&mut self, id: Id) {
    // let extract_start = Instant::now();
    // println!("---------------memo.size(): {}", self.memo.len());
    // if self.egraph[id].nodes.iter().any(|n| n.operation().is_lib()) {
    //   println!("eclass with lib, id: {id}");
    //   println!(
    //     "before eclass with lib extracted, best cost: {:?}",
    //     self.get_from_memo(id)
    //   );
    //   // println!("eclass: {:#?}", self.egraph[id].nodes);
    // }
    self.debug_indented(&format!("extracting eclass {id}"));
    if self.get_from_memo(id) == None {
      // Initialize memo with None to prevent infinite recursion in case of
      // cycles in the egraph
      self.insert_into_memo(id, None, None);
      // Extract a candidate expression from each node
      // println!("Extracting eclass {:#?}", self.egraph[id]);
      // let mut cnt = 0;
      for node in self.egraph[id].iter() {
        // println!("extracting node {}/{}", cnt, self.egraph[id].len());
        // cnt += 1;
        match self.extract_node(node) {
          None => (), // Extraction for this node failed (must be a cycle)
          Some(cand) => {
            // Extraction succeeded: check if cand is better than what we have
            // so far print!("eclss: {}, ", id);
            // print!("node: {}, ", node.operation());
            // print!("cand: {}, ", cand.pretty(100));
            // println!("cand gain: {}", Self::gain(&self, &cand));
            let mut flag = true;
            let mut cand_cost = Some(self.cost(&cand).cycles);
            let mut prev_msg = (0, None);
            let mut renew_flag = false;
            // 首先，如果self.get_from_memo(id) 有值，并且all_exprs中有值
            if let Some(index) = self.get_from_memo(id) {
              // println!("index: {}", index);
              if let Some(prev) = self.all_exprs[*index].clone() {
                // 存在表达式，首先计算prev的cost，
                // 如果all_costs中有值就不用计算，直接取出
                let prev_cost = if let Some(cost) = self.all_costs[*index] {
                  cost
                } else {
                  let cost = self.cost(&prev).cycles;
                  prev_msg = (index.clone(), Some(cost));
                  renew_flag = true;
                  cost
                };
                // 接下来计算cand的cost，并赋给cand_cost
                let c_cost = self.cost(&cand).cycles;
                // println!("id: {}", id);
                // println!("cand node: {:?}", node.operation());
                // println!("prev cost: {}, cand cost: {}", prev_cost, c_cost);
                // println!("");
                if prev_cost < c_cost {
                  flag = false;
                } else {
                  cand_cost = Some(c_cost);
                }
              }
            }
            if renew_flag {
              self.all_costs[prev_msg.0] = prev_msg.1;
            }
            // 如果flag为true，说明cand的cost更小,需要进行替换
            if flag {
              self.insert_into_memo(id, Some(cand.clone()), cand_cost);
            }
            // match self.get_from_memo(id) {
            //   // If we already had an expression and it was better, do
            // nothing   Some(index){
            //     // 首先检查all_cost中是否有值
            //     if let Some(cost) = self.all_costs[*index] {
            //       if cost < Self::cost(&self, &cand) {
            //         flag = false;
            //       }
            //     } else {
            //       // 如果没有值，检查
            //     }

            //   }
            // }
            //   // Otherwise, update the memo;
            //   // note that updating the memo after each better candidate is
            //   // found instead of at the end is slightly
            //   // suboptimal (because it might cause us to go around some
            // cycles   // once), but the code is simpler and it
            // doesn't   // matter too much.
            //   _ => {
            //     self.debug_indented(&format!(
            //       "new best for {id}: {} (gain {})",
            //       cand.pretty(100),
            //       Self::cost(&self, &cand)
            //     ));
            //     let start = Instant::now();
            //     self.insert_into_memo(id, Some(cand));
            //     println!(
            //       "using {}ms to insert into memo",
            //       start.elapsed().as_millis()
            //     );
            //   }
          }
        }
      }
      // if self.egraph[id].nodes.iter().any(|n| n.operation().is_lib()) {
      //   println!(
      //     "eclass with lib extracted, best cost: {:?}",
      //     self.get_from_memo(id)
      //   );
      //   // println!("eclass: {:#?}", self.egraph[id].nodes);
      // }
      // println!(
      //   "using {}ms to extract eclass {id}",
      //   extract_start.elapsed().as_millis()
      // );
      // println!("eclass id: {}, cost: {:?}", id, self.get_from_memo(id));
    }
  }

  /// Extract the smallest expression from `node`.
  fn extract_node(&mut self, node: &AstNode<Op>) -> MaybeExpr<Op> {
    self.debug_indented(&format!("extracting node {node:?}"));
    if let Some(BindingExpr::Lib(lid, _, _, _, _, _)) = node.as_binding_expr() {
      // println!("checking lib {lid}");
      if self.lib_context.contains(lid) {
        // This node is a definition of one of the libs, whose definition we are
        // currently extracting: do not go down this road since it leads
        // to lib definitions using themselves
        self.debug_indented(&format!("encountered banned lib: {lid}"));
        return None;
      }
    }
    // Otherwise: extract all children
    let mut child_indexes = vec![];
    // println!("extracting children of {node:?}");
    self.extract_children(node, 0, vec![], &mut child_indexes)
  }

  /// Process the children of `node` starting from index `current`
  /// and accumulate results in `partial expr`;
  /// `child_indexes` stores the indexes of already processed children within
  /// `partial_expr`, so that we can use them in the `AstNode` at the end.
  fn extract_children(
    &mut self,
    node: &AstNode<Op>,
    current: usize,
    mut partial_expr: Vec<AstNode<Op>>,
    child_indexes: &mut Vec<usize>,
  ) -> MaybeExpr<Op> {
    // let child_start = Instant::now();
    // println!("begin extracting children");
    if current == node.children().len() {
      // println!("current == children.len()");
      // Done with children: add ourselves to the partial expression and return
      let child_ids: Vec<Id> =
        child_indexes.iter().map(|x| (*x).into()).collect();
      let root = AstNode::new(node.operation().clone(), child_ids);
      partial_expr.push(root);
      // println!(
      //   "using {}ms to extract children",
      //   child_start.elapsed().as_millis()
      // );
      Some(partial_expr.into())
    } else {
      // If this is the first child of a lib node (i.e. lib definition) add this
      // lib to the context:
      let old_lib_context = self.lib_context.clone();
      if let Some(BindingExpr::Lib(lid, _, _, _, _, _)) = node.as_binding_expr()
      {
        if current == 0 {
          self.debug_indented(&format!(
            "processing first child of {node:?}, adding {lid} to context"
          ));
          self.lib_context.add(lid);
        }
      }

      // Process the current child
      let child = &node.children()[current];
      self.indent += 1;
      // println!(">>>extracting child {child:?}");
      self.extract(*child);
      self.indent -= 1;
      // We need to get the result before restoring the context
      // println!("begin getting child {child:?}");
      // let start = Instant::now();
      let child_res = self.get_from_memo(*child).clone();
      let child_res = if let Some(index) = child_res {
        self.all_exprs[*index].clone()
      } else {
        None
      };
      // println!("using {}ms to get the expr", start.elapsed().as_millis());
      // Restore lib context
      self.lib_context = old_lib_context;
      // println!("begin matching");
      match child_res {
        None => None, /* Failed to extract a child, so the extraction of */
        // this node fails
        Some(expr) => {
          // We need to clone the expr because we're going to mutate it (offset
          // child indexes), and we don't want it to affect the memo
          // result for child.
          let mut new_expr = expr.as_ref().to_vec();
          // println!("begin offsetting");
          for n in &mut new_expr {
            // Increment all indexes inside `n` by the current expression
            // length; this is needed to make a well-formed
            // `RecExpr`
            Self::offset_children(n, partial_expr.len());
          }
          // println!(">>>>> new expr");
          partial_expr.extend(new_expr);
          child_indexes.push(partial_expr.len() - 1);
          let exp = self.extract_children(
            node,
            current + 1,
            partial_expr,
            child_indexes,
          );
          // println!(">>>>>>> returning from child {child:?}");
          // println!(
          //   "using {}ms to extract children",
          //   child_start.elapsed().as_millis()
          // );
          exp
        }
      }
    }
  }

  /// Add `offset` to all children of `node`
  fn offset_children(node: &mut AstNode<Op>, offset: usize) {
    for child in node.children_mut() {
      let child_index: usize = (*child).into();
      *child = (child_index + offset).into();
    }
  }

  /// Print a debug message with the current indentation
  /// TODO: this should be a macro
  fn debug_indented(&self, msg: &str) {
    debug!("{:indent$}{msg}", "", indent = 2 * self.indent);
  }
}

// 定义trait TypeInfo
pub trait TypeInfo<T> {
  fn get_rtype(&self, child_types: &Vec<T>) -> T;
  fn merge_types(a: &T, b: &T) -> T;
  fn merge_types_neglecting_width(a: &T, b: &T) -> T;
  /// 这个函数是为了判断两个类型是否可以合并（判断标准是，
  /// 合并之后的硬件是不是可以作用在两个类型上）
  fn can_merge_types(a: &T, b: &T) -> bool;
}

// 定义GetType trait
pub trait ClassMatch {
  fn get_type(&self) -> Vec<String> {
    vec!["".to_string()]
  }
}

// 为ISAXCost实现ClassMatch trait
impl<T: PartialEq + Debug + Default + Clone + Ord + Hash + ToString> ClassMatch
  for ISAXCost<T>
{
  // fn type_match(&self, other: &Self) -> bool {
  //   self.ty == other.ty
  // }
  // fn level_match(&self, other: &Self) -> bool {
  //   let hash_similar =
  //     hamming_distance(self.hash.cls_hash, other.hash.cls_hash) < 36;
  //   let subtree_similar =
  //     jaccard_similarity(&self.hash.subtree_levels,
  // &other.hash.subtree_levels)
  //       > 0.67;
  //   hash_similar && subtree_similar
  // }
  fn get_type(&self) -> Vec<String> {
    vec![self.ty.to_string()]
  }
}

#[derive(Debug, Clone)]
pub struct ISAXCF<LA, LD> {
  bb_query: BBQuery,
  _phantom: PhantomData<(LA, LD)>,
}

impl<LA, LD> ISAXCF<LA, LD> {
  pub fn new(bb_query: BBQuery) -> Self {
    ISAXCF {
      bb_query,
      _phantom: PhantomData,
    }
  }
}

impl<Op, LA, LD> CostFunction<AstNode<Op>> for ISAXCF<LA, LD>
where
  Op: Clone
    + std::fmt::Debug
    + std::hash::Hash
    + Ord
    + Teachable
    + std::fmt::Display
    + OperationInfo,
  AstNode<Op>: Schedulable<LA, LD>,
{
  type Cost = OrderedFloat<f64>;

  fn cost<C>(&mut self, enode: &AstNode<Op>, costs: C) -> Self::Cost
  where
    C: FnMut(Id) -> Self::Cost,
  {
    OrderedFloat::from(0.0)
  }

  fn cost_rec(&mut self, expr: &RecExpr<AstNode<Op>>) -> Self::Cost {
    let (cycles, _) = rec_cost(expr, &self.bb_query, HashMap::new());
    let mut vec_num = 0;
    for node in expr.iter() {
      if node.operation().is_vector_op() {
        vec_num += 1;
      }
    }
    // 给vec_num一个奖励项，但又不能太大

    OrderedFloat::from(cycles) / (1.0 + (vec_num / expr.len()) as f64 * 0.01)
  }
}

#[derive(Debug, Clone)]
pub struct ISAXLpCF<LA, LD> {
  bb_query: BBQuery,
  _phantom: PhantomData<(LA, LD)>,
}

impl<LA, LD> ISAXLpCF<LA, LD> {
  pub fn new(bb_query: BBQuery) -> Self {
    ISAXLpCF {
      bb_query,
      _phantom: PhantomData,
    }
  }
}

pub fn eliminate_lambda<Op, N>(
  egraph: &EGraph<AstNode<Op>, N>,
) -> EGraph<AstNode<Op>, N>
where
  N: Analysis<AstNode<Op>> + Clone,
  N::Data: Clone,
  Op: OperationInfo + Ord + Debug + Clone + Hash + Teachable,
{
  let mut egraph_mut = egraph.clone();
  let mut visited = HashSet::new();
  for ecls in egraph.classes() {
    if ecls
      .nodes
      .iter()
      .any(|n| matches!(n.as_binding_expr(), Some(BindingExpr::Lambda(_))))
    {
      if visited.contains(&ecls.id) {
        continue;
      }
      // Recursively remove the lambda node and its children
      let mut stack = vec![ecls.id];
      // Remove the lambda node and its children
      while let Some(id) = stack.pop() {
        if visited.contains(&id) {
          continue;
        }
        visited.insert(id);
        let cur_ecls = &mut egraph_mut[id];
        for node in cur_ecls.nodes.iter_mut() {
          *node.operation_mut() =
            Op::make_rule_var(format!("Lambda{}", cur_ecls.id));
          for child in node.children() {
            stack.push(*child);
          }
        }
      }
    }
  }
  // egraph_mut.rebuild();
  egraph_mut
}

impl<LA, LD, Op, T> LpCostFunction<AstNode<Op>, ISAXAnalysis<Op, T>>
  for ISAXLpCF<LA, LD>
where
  AstNode<Op>: Schedulable<LA, LD>,
  Op: Clone
    + std::fmt::Debug
    + std::hash::Hash
    + Ord
    + Teachable
    + std::fmt::Display
    + OperationInfo
    + Arity
    + Send
    + Sync
    + 'static,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  fn node_cost(
    &mut self,
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
    _eclass: Id,
    enode: &AstNode<Op>,
  ) -> f64 {
    let bbs = enode.operation().get_bbs_info();
    if bbs.is_empty() {
      return 0.0; // No BBS, no cost
    }

    let analysis = egraph.analysis.clone();
    let exe_count = enode.operation().op_execution_count(&self.bb_query);
    if let Some(BindingExpr::Lib(lib_id, _, _, _, lat_acc, _)) =
      enode.as_binding_expr()
    {
      let lib_id = lib_id.0;
      let mut bbs = enode.operation().get_bbs_info();
      bbs.sort();
      let extract_lat_acc = analysis.get_lat_acc(lib_id, bbs.clone());
      let lat_acc = if let Some(lat) = extract_lat_acc {
        lat
      } else {
        // If we don't have latency for this lib, use the default
        // latency from the binding expr
        lat_acc.0
      };
      lat_acc * exe_count
    } else if enode.operation().is_op() {
      enode.op_latency_cpu(&self.bb_query) * exe_count
    } else {
      0.0 // This is a no-op, so no cost
    }
  }
}

#[derive(Debug)]
pub struct ISAXExtractor<
  'a,
  Op: Clone
    + std::fmt::Debug
    + std::hash::Hash
    + Ord
    + Teachable
    + std::fmt::Display
    + OperationInfo,
  N: Analysis<AstNode<Op>>,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  LA,
  LD,
> where
  AstNode<Op>: TypeInfo<T> + Schedulable<LA, LD>,
{
  best_nodes: FxHashMap<Id, (usize, OrderedFloat<f64>)>,
  /// The egraph to extract from.
  egraph: &'a EGraph<AstNode<Op>, N>,
  bb_query: BBQuery,
  _phantom: PhantomData<(T, LA, LD)>,
}

impl<'a, Op, N, T, LA, LD> ISAXExtractor<'a, Op, N, T, LA, LD>
where
  Op: Clone
    + std::fmt::Debug
    + std::hash::Hash
    + Ord
    + Teachable
    + std::fmt::Display
    + OperationInfo,
  N: Analysis<AstNode<Op>> + Clone,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T> + Schedulable<LA, LD>,
{
  pub fn new(egraph: &'a EGraph<AstNode<Op>, N>, bb_query: BBQuery) -> Self {
    ISAXExtractor {
      best_nodes: FxHashMap::default(),
      egraph,
      bb_query,
      _phantom: PhantomData,
    }
  }

  fn expr_from_id(&self, id: Id) -> RecExpr<AstNode<Op>> {
    let mut expr: Vec<AstNode<Op>> = Vec::new();
    let mut visited = HashMap::new();
    self.best_expr(id, &mut expr, &mut visited);
    expr.into()
  }

  fn best_expr(
    &self,
    id: Id,
    expr: &mut Vec<AstNode<Op>>,
    visited: &mut HashMap<Id, usize>,
  ) {
    if visited.contains_key(&id) {
      return;
    }

    let (index, _) = self.best_nodes.get(&id).unwrap();
    let mut node = self.egraph[id].nodes[*index].clone();
    let mut args_idx = vec![];
    for child in node.children() {
      self.best_expr(*child, expr, visited);
      args_idx.push(visited.get(&child).unwrap().clone());
    }

    let args_len = node.children().len();
    let args_mut = node.args_mut();
    for i in 0..args_len {
      args_mut[i] = Id::from(args_idx[i]);
    }

    visited.insert(id, expr.len());
    expr.push(node);
  }

  pub fn best(&mut self, id: Id) -> RecExpr<AstNode<Op>> {
    // println!("extracting eclass {id}");
    self.extract(id);
    // println!("id: {:#?}", id);
    // println!("{:#?}", self.egraph[id]);
    // Get the best expression from the memo:
    let expr = self.expr_from_id(id);
    // println!("best expr: {expr:?}");
    expr
  }

  fn cost(&self, expr: &RecExpr<AstNode<Op>>) -> OrderedFloat<f64> {
    let (cycles, _) = rec_cost(expr, &self.bb_query, HashMap::new());
    // println!("cycles: {}", cycles);
    OrderedFloat::from(cycles)
  }

  pub fn extract(&mut self, id: Id) {
    if let None = self.best_nodes.get(&id) {
      // println!("extracting eclass {id}");
      self
        .best_nodes
        .insert(id, (0, OrderedFloat::from(f64::MAX)));
      let ecls = &self.egraph[id];
      for (idx, node) in ecls.nodes.iter().enumerate() {
        let best_node = self.best_nodes.get_mut(&id).unwrap();
        let cur_best = *best_node;
        *best_node = (idx, OrderedFloat::from(f64::MAX));

        for child in node.children() {
          // println!("extracting child {child:?}");
          self.extract(*child);
        }

        if ecls.nodes.len() > 1 {
          if idx == 0 {
            let expr = self.expr_from_id(id);
            let cost = self.cost(&expr);
            // println!("extracting eclass {id}, cost: {cost}");
            let best_node = self.best_nodes.get_mut(&id).unwrap();
            *best_node = (idx, cost);
          } else {
            // Compare the current node with the previous best
            let expr = self.expr_from_id(id);
            let cost = self.cost(&expr);
            // println!("extracting eclass {id}, cost: {cost}");
            // println!("current best: {}", cur_best.1);
            // println!("cost of node {idx}: {cost}");
            let best_node = self.best_nodes.get_mut(&id).unwrap();
            if cost < cur_best.1 {
              *best_node = (idx, cost);
            } else {
              // println!("not better than previous best: {cur_best:?}");
              *best_node = cur_best;
            }
          }
        }
      }
    }
  }
}
