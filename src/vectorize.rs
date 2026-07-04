use crate::{
  BindingExpr, DeBruijnIndex, DiscriminantEq, Expr, LibId, PartialExpr,
  Printable,
  analysis::SimpleAnalysis,
  ast_node::{Arity, AstNode},
  au_filter::TypeAnalysis,
  bb_query::{self, BBInfo, BBQuery},
  extract::beam_pareto::{
    ClassMatch, ISAXAnalysis, ISAXCF, ISAXCost, ISAXLpCF, TypeInfo,
  },
  learn::{LearnedLibraryBuilder, LiblearnMessage, reify},
  perf_infer,
  rewrites::TypeMatch,
  runner::{
    self, AUMergeMod, EnumMode, LiblearnConfig, LiblearnCost, OperationInfo,
    ParetoConfig, ParetoResult,
  },
  schedule::{Schedulable, Scheduler, rec_cost},
  teachable::{ShieldingOp, Teachable},
};

use bitvec::vec;
use egg::{
  Analysis, AstDepth, AstSize, ConditionalApplier, CostFunction, EClass,
  EGraph, Extractor, Id, Language, LpCostFunction, LpExtractor, Pattern,
  RecExpr, Rewrite, Runner, Searcher, Subst, Var, rewrite,
};

use indexmap::IndexSet;
use lexpr::print;
use log::debug;

use nom::lib;
use serde::Deserialize;
use std::{
  cmp::{self, Ordering},
  collections::{BTreeMap, HashMap, HashSet},
  convert::Infallible,
  fmt::{Debug, Display},
  hash::{Hash, Hasher},
  str::FromStr,
  sync::mpsc,
  time::Duration,
};

/// 用于进行向量化的config
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct VectorConfig {
  /// 是否进行向量化, 不进行序列化，这个参数是由PhaseConfig控制的
  #[serde(skip)]
  pub vectorize: bool,
  /// 是否启用gather节点(gather节点搜索数目较少， 对整体速度的影响不是很大，
  /// 默认为true)
  pub enable_gather: bool,
  /// 是否启用shuffle节点(shuffle节点搜索数目很大， 对整体速度的影响较大，
  /// 默认为false， 但是加入shuffle节点之后， 向量化的性能会很高)
  pub enable_shuffle: bool,
  /// 是否启用post-check(在预处理阶段，
  /// 对于store和load节点的深嵌套进行了解耦，但是可能导致错误的结果)
  pub enable_post_check: bool,
  /// 最大允许的向量化组大小、
  pub max_vec_len: usize,
  /// lift-rules的文件名
  pub lift_rules: Option<String>,
  /// lower-rules的文件名
  pub lower_rules: Option<String>,
  /// transfrom-rules的文件名
  pub transform_rules: Option<String>,
}

impl Default for VectorConfig {
  fn default() -> Self {
    Self {
      vectorize: false,
      enable_gather: true,
      enable_shuffle: false,
      enable_post_check: false,
      max_vec_len: 8,
      lift_rules: None,
      lower_rules: None,
      transform_rules: None,
    }
  }
}

impl VectorConfig {
  /// Create a new VectorConfig with the given parameters
  pub fn new(
    vectorize: bool,
    enable_gather: bool,
    enable_shuffle: bool,
    enable_post_check: bool,
    max_vec_len: usize,
    lift_rules: Option<String>,
    lower_rules: Option<String>,
    transform_rules: Option<String>,
  ) -> Self {
    Self {
      vectorize,
      enable_gather,
      enable_shuffle,
      enable_post_check,
      max_vec_len,
      lift_rules,
      lower_rules,
      transform_rules,
    }
  }
}

/// 最大化向量化成本模型：鼓励更长的向量操作。
///
/// 核心思路：
/// - 对所有操作按其基础代价 `base_cost` 进行计算（由 `get_simple_cost` 返回）
/// - 如果是向量操作，则对 `base_cost` 应用长度奖励： `adjusted_cost = base_cost
///   / (1.0 + alpha * vec_len)`，向量越长，折算后成本越低。
/// - 否则使用 `base_cost + scalar_penalty`，对标量操作增加惩罚。
/// - 子节点代价合并：
///   - 向量节点取子节点成本的最大值（突出最长路径瓶颈）。
///   - 标量节点累加所有子节点成本（反映总成本）。

#[derive(Debug, Clone)]
pub struct MaxVectorCF {
  /// 奖励系数，向量越长奖励越大
  pub alpha: f64,
  /// 标量操作的额外惩罚
  pub scalar_penalty: f64,
}

impl MaxVectorCF {
  /// 创建新的 MaxVectorCF
  pub fn new(alpha: f64, scalar_penalty: f64) -> Self {
    Self {
      alpha,
      scalar_penalty,
    }
  }
}

impl<Op> CostFunction<AstNode<Op>> for MaxVectorCF
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Send
    + Sync
    + Debug
    + OperationInfo,
{
  type Cost = f64;

  fn cost<C>(&mut self, enode: &AstNode<Op>, mut costs: C) -> Self::Cost
  where
    C: FnMut(Id) -> Self::Cost,
  {
    // 基础代价
    let mut base = enode.operation().get_simple_cost();

    // 计算向量长度
    let vec_len = if enode.operation().is_vec() {
      enode.args().len()
    } else if enode.operation().is_vector_op() {
      enode.operation().get_vec_len()
    } else {
      0
    };

    // 根据 vec_len 调整或惩罚基础代价
    if vec_len > 0 {
      // 向量操作：奖励更长的向量
      base = base / (1.0 + self.alpha * vec_len as f64);
    } else {
      // 标量操作：加惩罚
      base += self.scalar_penalty;
    }

    // 合并子节点成本
    let children_cost: f64 = if enode.operation().is_vec() {
      // 向量节点：取最大子成本，突显最长瓶颈路径
      enode.args().iter().map(|&id| costs(id)).fold(0.0, f64::max)
    } else {
      // 标量节点：累加所有子成本
      enode.args().iter().map(|&id| costs(id)).sum()
    };

    base + children_cost
  }
}

/// 为 MaxVectorCF 实现 LpCostFunction，在 LP 模式下只取节点自身的调整代价
#[cfg_attr(docsrs, doc(cfg(feature = "lp")))]
impl<Op, T> LpCostFunction<AstNode<Op>, ISAXAnalysis<Op, T>> for MaxVectorCF
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Send
    + Sync
    + Debug
    + 'static
    + DiscriminantEq
    + Default
    + Printable
    + OperationInfo,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + FromStr
    + 'static,
  AstNode<Op>: TypeInfo<T>,
{
  fn node_cost(
    &mut self,
    _egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
    _eclass: Id,
    enode: &AstNode<Op>,
  ) -> f64 {
    let op = enode.operation();
    // 基础代价
    let base = op.get_simple_cost();
    // 计算向量长度
    let vec_len = if op.is_vec() {
      // 假设 Language 上的方法提供 args
      enode.args().len()
    } else if op.is_vector_op() {
      op.get_vec_len()
    } else {
      0
    };
    // 应用奖励或惩罚
    if vec_len > 0 {
      base / (1.0 + self.alpha * vec_len as f64)
    } else {
      base + self.scalar_penalty
    }
  }
}

pub fn find_vec_containments_and_gathers<Op, T>(
  egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
) -> (HashMap<Id, HashSet<Id>>, HashMap<(Id, Id), Vec<usize>>)
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + BBInfo
    + Arity
    + Send
    + Sync
    + Debug
    + 'static
    + DiscriminantEq
    + Default
    + Printable
    + OperationInfo,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + FromStr
    + 'static,
  AstNode<Op>: TypeInfo<T>,
{
  // 收集所有 vec 元素列表，按长度分组
  let mut len_groups: BTreeMap<usize, Vec<(Id, Vec<Id>, HashSet<Id>)>> =
    BTreeMap::new();

  for eclass in egraph.classes() {
    if let Some(enode) = eclass.nodes.iter().find(|n| n.operation().is_vec()) {
      let elements_vec: Vec<Id> = enode.args().to_vec();
      let elements_set: HashSet<Id> = elements_vec.iter().cloned().collect();
      let len = elements_vec.len();
      len_groups.entry(len).or_default().push((
        eclass.id,
        elements_vec,
        elements_set,
      ));
    }
  }

  let mut direct_map: HashMap<Id, HashSet<Id>> = HashMap::new();
  let mut gather_map: HashMap<(Id, Id), Vec<usize>> = HashMap::new();

  for (current_len, current_vecs) in &len_groups {
    for (_higher_len, higher_vecs) in len_groups.range(current_len + 1..) {
      for (a_id, a_vec, a_set) in current_vecs {
        for (b_id, b_vec, b_set) in higher_vecs {
          if a_set.is_subset(b_set) {
            // 更新包含关系
            direct_map.entry(*a_id).or_default().insert(*b_id);

            // 计算 gather 索引
            let mut indices = Vec::new();
            for a_elem in a_vec {
              if let Some(pos) =
                b_vec.iter().position(|b_elem| b_elem == a_elem)
              {
                indices.push(pos);
              } else {
                panic!("Element from A not found in B, inconsistent state");
              }
            }
            gather_map.insert((*a_id, *b_id), indices);
          }
        }
      }
    }
  }

  (direct_map, gather_map)
}

pub fn find_vec_shuffles_and_indices<Op, T>(
  egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
) -> (
  HashMap<Id, HashSet<(Id, Id)>>, /* For each C, which (A,B) pairs can
                                   * shuffle into C */
  HashMap<(Id, Id, Id), Vec<usize>>, /* For each (C, A, B), the index‐vector
                                      * to gather from [A; B] to C */
)
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + BBInfo
    + Arity
    + Send
    + Sync
    + Debug
    + 'static
    + DiscriminantEq
    + Default
    + Printable
    + OperationInfo,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + FromStr
    + 'static,
  AstNode<Op>: TypeInfo<T>,
{
  // 1) 收集所有 vec 表达式，按长度分组
  let mut len_groups: BTreeMap<usize, Vec<(Id, Vec<Id>, HashSet<Id>)>> =
    BTreeMap::new();

  for eclass in egraph.classes() {
    if let Some(enode) = eclass.nodes.iter().find(|n| n.operation().is_vec()) {
      let elems: Vec<Id> = enode.args().to_vec();
      let set: HashSet<Id> = elems.iter().cloned().collect();
      len_groups
        .entry(elems.len())
        .or_default()
        .push((eclass.id, elems, set));
    }
  }

  // 2) 准备输出容器
  let mut shuffle_pairs: HashMap<Id, HashSet<(Id, Id)>> = HashMap::new();
  let mut shuffle_map: HashMap<(Id, Id, Id), Vec<usize>> = HashMap::new();

  // 3) 对每一对 (A, B) 组合尝试所有可能的 C 假设 m + n = L，即 A.len() +
  //    B.len() == C.len()
  let lengths: Vec<usize> = len_groups.keys().cloned().collect();
  for &len_a in &lengths {
    for &len_b in &lengths {
      let total_len = len_a + len_b;
      // 对每个具体的 A、B vec
      for (a_id, a_vec, a_set) in &len_groups[&len_a] {
        for (b_id, b_vec, b_set) in &len_groups[&len_b] {
          if a_id > b_id {
            continue; // 避免重复处理 (A, B) 和 (B, A)
          }
          // 预先构造 [A; B] 的向量和并集
          let mut ab_vec = Vec::with_capacity(total_len);
          ab_vec.extend(a_vec.iter().cloned());
          ab_vec.extend(b_vec.iter().cloned());
          let union_set: HashSet<Id> = a_set.union(b_set).cloned().collect();

          // 只考虑 C 长度为 total_len 的候选
          for (&len_c, c_group) in &len_groups {
            if len_c != total_len {
              continue; // Skip if C's length doesn't match A+B length
            }
            for (c_id, c_vec, c_set) in c_group {
              // 若 C 的元素集合是 A∪B 的子集，则可能是 shuffle
              if !c_set.is_subset(&union_set) {
                continue;
              }
              // 计算从 [A; B] gather 出 C 所需的索引
              let mut indices = Vec::with_capacity(c_vec.len());
              for &elem in c_vec {
                if let Some(pos) = ab_vec.iter().position(|&x| x == elem) {
                  indices.push(pos);
                } else {
                  // 逻辑上不应发生：C_set ⊆ union_set 已保证
                  panic!(
                    "Internal error: element {:?} of C={:?} \
                                       not found in concatenated A={:?},B={:?}",
                    elem, c_id, a_id, b_id
                  );
                }
              }
              // 记录这一对 (A, B) 能生成 C
              shuffle_pairs
                .entry(*c_id)
                .or_default()
                .insert((*a_id, *b_id));
              shuffle_map.insert((*c_id, *a_id, *b_id), indices);
            }
          }
        }
      }
    }
  }

  (shuffle_pairs, shuffle_map)
}

pub struct ReachCache {
  children: HashMap<Id, Vec<Id>>,
  cache: HashMap<(Id, Id), bool>,
}

impl ReachCache {
  pub fn new<'a, Op, T>(
    egraph: &'a EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  ) -> Self
  where
    Op: std::fmt::Debug
      + std::hash::Hash
      + Clone
      + Ord
      + Teachable
      + BBInfo
      + Arity
      + Send
      + Sync
      + std::fmt::Display
      + 'static
      + DiscriminantEq
      + Default
      + Printable
      + OperationInfo,
    T: std::fmt::Debug
      + Default
      + Clone
      + PartialEq
      + Ord
      + std::hash::Hash
      + Send
      + Sync
      + std::fmt::Display
      + std::str::FromStr
      + 'static,
    AstNode<Op>: TypeInfo<T>,
  {
    let mut children: HashMap<Id, Vec<Id>> = HashMap::new();
    // Only consider a child eclass when all enodes in the parent class depend
    // on it
    // for class in egraph.classes() {
    //   // println!("ecls: {}, nodes: {:?}", class.id, class.nodes);
    //   let enodes = &egraph[class.id].nodes;
    //   let total = enodes.len();
    //   // count occurrences per child id across all enodes
    //   let mut count_map: HashMap<Id, usize> = HashMap::new();
    //   for enode in enodes {
    //     // track unique children per enode to avoid double counting
    //     let mut seen = HashSet::new();
    //     for &child_id in enode.children() {
    //       if seen.insert(child_id) {
    //         *count_map.entry(child_id).or_default() += 1;
    //       }
    //     }
    //   }
    //   // include only those child_ids seen in every enode
    //   let common_children = count_map
    //     .into_iter()
    //     .filter_map(
    //       |(child_id, cnt)| if cnt == total { Some(child_id) } else { None },
    //     )
    //     .collect::<Vec<_>>();
    //   if !common_children.is_empty() {
    //     children.insert(class.id, common_children);
    //   }
    // }
    // 目前考虑任意的子节点
    for class in egraph.classes() {
      // println!("ecls: {}, nodes: {:?}", class.id, class.nodes);
      let enodes = &egraph[class.id].nodes;
      let children_ids = enodes
        .into_iter()
        .flat_map(|enode| enode.args())
        .map(|&child_id| child_id)
        .collect::<Vec<_>>();
      if !children_ids.is_empty() {
        children.insert(class.id, children_ids);
      }
    }
    Self {
      children,
      cache: HashMap::new(),
    }
  }

  /// 判断 src 是否能 reach dst，带 cache
  fn reachable(&mut self, src: Id, dst: Id) -> bool {
    if let Some(&cached) = self.cache.get(&(src, dst)) {
      return cached;
    }

    let mut visited = HashSet::new();
    let mut stack = vec![src];
    let mut found = false;

    while let Some(cur) = stack.pop() {
      if cur == dst {
        found = true;
        break;
      }
      if !visited.insert(cur) {
        continue;
      }
      if let Some(children) = self.children.get(&cur) {
        for &child in children {
          stack.push(child);
        }
      }
    }

    self.cache.insert((src, dst), found);
    found
  }

  /// 判断给定 group 是否内部无依赖（可打包）
  pub fn can_pack(&mut self, group: &[Id]) -> bool {
    for (i, &a) in group.iter().enumerate() {
      for &b in &group[i + 1..] {
        if self.reachable(a, b) || self.reachable(b, a) {
          // println!("Cannot pack {:?} and {:?} due to reachability", a, b);
          return false;
        }
      }
    }
    true
  }
}

fn add_packs<Op, T>(
  egraph: &mut EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  id_set: &mut HashSet<Vec<Id>>,
  reach_cache: &mut ReachCache,
  packs: &HashMap<Vec<Id>, Vec<String>>,
) where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + BBInfo
    + Arity
    + Send
    + Sync
    + Debug
    + 'static
    + DiscriminantEq
    + Default
    + Printable
    + OperationInfo,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + FromStr
    + 'static,
  AstNode<Op>: TypeInfo<T>,
{
  for (id_pack, bbs) in packs.into_iter() {
    // 如果 id_pack 已经存在，直接跳过
    if id_set.contains(id_pack) {
      // println!("id_pack already exists, skip");
      continue;
    }
    // 如果id_pack不能pack，直接跳过
    if !reach_cache.can_pack(&id_pack) {
      // println!("id_pack cannot pack, skip");
      continue;
    }
    id_set.insert(id_pack.clone());
    // println!("id_pack: {:?}", id_pack);
    // println!("found {} matches for {:?}", results.len(), rewrite);
    let mut tys = Vec::new();
    for &matched_eclass_id in id_pack {
      // 获取每个eclass的类型信息
      let ty = egraph[matched_eclass_id].data.ty.to_string();
      tys.push(ty);
    }
    let vec_node =
      AstNode::new(Op::make_vec(tys.clone(), bbs.clone()), id_pack.clone());

    let list_id = egraph.add(vec_node);

    // 针对每个id，构造一个get节点
    for i in 0..id_pack.len() {
      let get_node = AstNode::new(
        Op::make_get_from_vec(i, vec![tys[i].clone()], bbs.clone()),
        vec![list_id],
      );
      // println!("tys[i]: {:?}", tys[i]);
      let get_id = egraph.add(get_node);
      // println!("get_eclass: {:?}", egraph[get_id]);
      // 进行替换
      egraph.union(id_pack[i], get_id);
      // println!("success union");
    }
    egraph.rebuild();
  }
}

/// 目前向量化直接使用liblearn中的au-search进行向量化
pub fn vectorize<Op, T, LA, LD>(
  egraph_running_scalar_dsrs: EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  root: Id,
  lift_dsrs: &Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  lower_dsrs: &Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  transfrom_dsrs: &Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  config: ParetoConfig<LA, LD>,
  bb_query: BBQuery,
  max_lib_id: usize,
) -> (
  (EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>, Id),
  Vec<LiblearnMessage<Op, T, ISAXAnalysis<Op, T>>>,
)
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + BBInfo
    + Arity
    + Send
    + Sync
    + Debug
    + 'static
    + DiscriminantEq
    + Default
    + Printable
    + OperationInfo,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + FromStr
    + 'static
    + TypeAnalysis,
  LA: Debug + Clone + Default,
  LD: Debug + Clone + Default,
  AstNode<Op>: TypeInfo<T> + Schedulable<LA, LD>,
{
  println!("        • there are {} lift dsrs", lift_dsrs.len());
  println!("        • there are {} lower dsrs", lower_dsrs.len());
  println!(
    "        • there are {} transfrom dsrs",
    transfrom_dsrs.len()
  );
  let timeout = Duration::from_secs(60 * 100_000);
  let mut egraph = egraph_running_scalar_dsrs.clone();
  println!(
    "       • before vectorize, eclass size: {}, egraph size: {}",
    egraph.classes().len(),
    egraph.total_size()
  );
  // 第一步，拿到list_packs
  // list节点的子节点也需要加入，只要子节点的bbs信息相同
  let mut id_set = HashSet::new();
  let mut reach_cache = ReachCache::new(&egraph);
  let mut list_packs: HashMap<Vec<Id>, Vec<String>> = HashMap::new();
  for eclass in egraph.classes() {
    for node in eclass.nodes.iter() {
      if node.operation().is_lib() {
        // 获取子节点的bbs信息
        let bbs = eclass.data.bb.clone();
        // 获取子节点的id
        let ids = node.args().to_vec();
        if ids.len() < 2 {
          continue; // 至少需要两个子节点
        }
        // 将ids和bbs信息加入到list_packs中
        list_packs.insert(ids, bbs);
      }
    }
  }
  add_packs(&mut egraph, &mut id_set, &mut reach_cache, &list_packs);
  let vetorize_lib_config = LiblearnConfig::new(
    LiblearnCost::Size,
    AUMergeMod::Boundary,
    EnumMode::PruningGold,
    // 后面的配置直接使用config.liblearn中的配置
    config.liblearn_config.sample_num,
    config.liblearn_config.hamming_threshold,
    config.liblearn_config.jaccard_threshold,
    config.liblearn_config.max_libs,
    config.liblearn_config.min_lib_size,
    config.liblearn_config.max_lib_size,
  );

  // 推导bbs信息
  perf_infer::perf_infer(&mut egraph, &vec![root]);

  // // 进行向量化组发现
  // egraph.dot().to_png("target/foo1.png").unwrap();
  // 进行库学习
  let mut learned_lib = LearnedLibraryBuilder::default()
    .learn_constants(config.learn_constants)
    .max_arity(config.max_arity)
    // .with_co_occurs(co_occurs)
    .with_last_lib_id(0)
    .with_liblearn_config(vetorize_lib_config)
    .with_clock_period(config.clock_period)
    .with_area_estimator(config.area_estimator.clone())
    .with_delay_estimator(config.delay_estimator.clone())
    .find_packs()
    .with_find_pack_config(config.find_pack_config.clone())
    .with_bb_query(bb_query.clone())
    .build(&egraph, vec![root]);

  learned_lib.delete_libs_according_size();
  let mut lib_rewrites: HashMap<
    usize,
    Vec<(Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>, Pattern<_>)>,
  > = HashMap::new();
  for m in learned_lib.messages() {
    lib_rewrites
      .entry(m.lib_id)
      .or_default()
      .push((m.rewrite.clone(), Pattern::from(m.searcher_pe.clone())));
  }
  println!(
    "        • Vectorization::learned {} libs",
    lib_rewrites.len()
  );
  // println!("        • Vectorization::lib_ids: {}", lib_ids.len());
  // for lib in learned_lib.libs() {
  //   println!("lib: {}", lib);
  // }
  let egraph_clone = egraph.clone();

  for (id, rewrites) in lib_rewrites.clone().into_iter() {
    // let rewrite = rewrite.0.clone();
    let (tx, rx) = mpsc::channel();
    let egraph_clone = egraph_clone.clone();
    let searchers = rewrites
      .iter()
      .map(|(_, searcher)| searcher.clone())
      .collect::<Vec<_>>();
    std::thread::spawn(move || {
      let results: Vec<egg::SearchMatches<'_, AstNode<Op>>> = searchers
        .iter()
        .flat_map(|searcher| searcher.search(&egraph_clone))
        .collect();
      let classes = results.iter().map(|x| x.eclass).collect::<Vec<_>>();
      let _ = tx.send(classes);
    });

    match rx.recv_timeout(Duration::from_secs(5)) {
      Ok(results) => {
        if results.len() < 2 {
          // 如果结果小于2，直接跳过
          // println!("I found {} results, skip", results.len());
          continue;
        }
        // 需要根据每个eclass的bbs信息进行分类
        let bbs_packs: HashMap<Vec<String>, Vec<Id>> =
          results.iter().fold(HashMap::new(), |mut acc, eclass| {
            let bbs = egraph[*eclass].data.bb.clone();
            // println!("child_id: {}, bbs: {:?}", eclass, bbs);
            acc.entry(bbs.clone()).or_default().push(*eclass);
            acc
          });
        let mut fine_packs: HashMap<Vec<Id>, Vec<String>> = HashMap::new();

        for (bbs, ids) in bbs_packs.iter() {
          // 排序去重
          let mut sorted_ids = ids.clone();
          sorted_ids.sort();
          sorted_ids.dedup();

          let mut current = Vec::new();
          for &id in &sorted_ids {
            let mut trial = current.clone();
            trial.push(id);

            if trial.len() <= 8 && reach_cache.can_pack(&trial) {
              // 可以扩包
              current.push(id);
            } else {
              // 达到上限或达不到 pack 条件，先收录已有的 current（需至少 2
              // 个元素）
              if current.len() >= 2 {
                fine_packs.insert(current.clone(), bbs.clone());
              }
              // 不足 2 的就直接丢弃掉（或根据需求另行处理）
              // 开启新包
              current = vec![id];
            }
          }
          // 处理最后剩下的 current
          if current.len() >= 2 {
            fine_packs.insert(current.clone(), bbs.clone());
          }
        }
        add_packs(&mut egraph, &mut id_set, &mut reach_cache, &fine_packs);

        // 重建egraph
      }
      Err(_) => {
        eprintln!("rewrite {} timed out, skip", id);
        // 直接继续下一个，不用等子线程
        continue;
      }
    }
  }

  // egraph.dot().to_png("target/foo1.png").unwrap();
  println!(
    "       • after add list, egraph size: {}, class size: {}",
    egraph.total_size(),
    egraph.classes().len()
  );

  //使用dsrs进行重写
  let runner = Runner::<_, _, ()>::new(ISAXAnalysis::empty())
    .with_egraph(egraph.clone())
    .with_time_limit(timeout)
    .with_iter_limit(4)
    .run(lift_dsrs);

  // 目前已经实现了各类vec的构建，现在需要去寻找egraph中含有的vec
  // enode，然后加入gather节点
  let mut egraph = runner.egraph.clone();
  // 首先，将eclass中的bbs信息加入空bbs的enode
  for class in egraph.classes_mut() {
    for node in class.nodes.iter_mut() {
      // 如果node的bbs信息是空的，就将eclass的bbs信息传给它
      if node.operation().get_bbs_info().is_empty() {
        let bbs = class.data.bb.clone();
        node.operation_mut().set_bbs_info(bbs);
      }
    }
  }
  // 对于vec节点而言，如果其bbs信息是空的，就需要保持和子节点一致
  for eclass in egraph.classes_mut() {
    for node in eclass.nodes.iter_mut() {
      if node.operation().is_vec() && node.operation().get_bbs_info().is_empty()
      {
        let mut bbs = HashSet::new();
        for arg in node.args() {
          // 否则就将arg的bbs信息加入到bbs中
          bbs.extend(runner.egraph[*arg].data.bb.clone());
        }
        let mut bbs = bbs.into_iter().collect::<Vec<_>>();
        bbs.sort();

        node.operation_mut().set_bbs_info(bbs.clone());
      }
    }
  }
  egraph.rebuild();

  perf_infer::perf_infer(&mut egraph, &vec![root]);

  // egraph.dot().to_png("target/foo2.png").unwrap();
  if config.vectorize_config.enable_gather {
    let (vec_containments, vec_gathers) =
      find_vec_containments_and_gathers(&egraph);
    println!(
      "        • size of vec_containments: {}",
      vec_containments.len()
    );
    // 根据包含关系加入gather节点
    for (i, (id, fathers)) in vec_containments.into_iter().enumerate() {
      // println!("i:{}, id: {}, fathers: {}", i, id, fathers.len());
      for father in fathers {
        if egraph[id].data.bb != egraph[father].data.bb {
          // 如果id和father的bbs不一致，直接跳过
          continue;
        }
        let bbs = egraph[id].data.bb.clone();
        let gather_op = Op::make_gather(&vec_gathers[&(id, father)], bbs);
        let cani_father = egraph.find(father);
        let gather_node = AstNode::new(gather_op.clone(), vec![cani_father]);
        let gather_id = egraph.add(gather_node);
        // 进行替换
        let cani_id = egraph.find(id);
        egraph.union(cani_id, gather_id);
      }
    }
    egraph.rebuild();
  }
  if config.vectorize_config.enable_shuffle {
    // 除此之外，还需要加入shuffle节点
    let (shuffle_pairs, shuffle_map) = find_vec_shuffles_and_indices(&egraph);
    println!("        • size of shuffle_pairs: {}", shuffle_pairs.len());
    // egraph.dot().to_png("target/foo1.png").unwrap();
    // 根据 shuffle 关系加入 shuffle 节点
    for (c_id, pairs) in shuffle_pairs.into_iter() {
      // println!("c_id: {}, pairs: {}", c_id, pairs.len());
      for (a_id, b_id) in pairs {
        // 检查 A 和 B 的 bbs 是否一致
        if egraph[a_id].data.bb != egraph[b_id].data.bb {
          // 如果 A 和 B 的 bbs 不一致，直接跳过
          continue;
        }
        // 检查 C 的 bbs 是否一致
        if egraph[c_id].data.bb != egraph[a_id].data.bb {
          // 如果 C 的 bbs 不一致，直接跳过
          continue;
        }
        let bbs = egraph[c_id].data.bb.clone();
        // 从 shuffle_map 中获取 (C, A, B) 对应的 gather 索引
        let indices = &shuffle_map[&(c_id, a_id, b_id)];

        // 查找 A 和 B 的父节点
        let a_parent = egraph.find(a_id);
        let b_parent = egraph.find(b_id);

        // 创建 shuffle 操作节点
        let shuffle_op = Op::make_shuffle(indices, bbs); // 这里假设 `make_shuffle`
        // 是定义的函数
        let shuffle_node =
          AstNode::new(shuffle_op.clone(), vec![a_parent, b_parent]);

        // 添加 shuffle 节点到 egraph
        let shuffle_id = egraph.add(shuffle_node);

        // 进行替换
        let cani_id = egraph.find(c_id);
        egraph.union(cani_id, shuffle_id);
      }
    }
    // 重新构建 egraph
    egraph.rebuild();
  }

  // egraph.dot().to_png("target/foo.png").unwrap();
  println!(
    "       • after add vectors, egraph size: {}, class size: {}",
    egraph.total_size(),
    egraph.classes().len()
  );

  // 使用transform_dsrs进行重写
  // let runner = Runner::<_, _, ()>::new(ISAXAnalysis::empty())
  //   .with_egraph(egraph)
  //   .with_time_limit(timeout)
  //   .with_iter_limit(4)
  //   .run(transfrom_dsrs);
  // egraph = runner.egraph.clone();

  // 恢复类型信息，将每个eclass的类型信息传给里面的enode

  let cloned_egraph = egraph.clone();
  for class in egraph.classes_mut() {
    for node in class.nodes.iter_mut() {
      // 如果node的ty是空的，就将eclass的类型信息传给它
      if node.operation().get_result_type().is_empty() {
        let tys = cloned_egraph[class.id].data.get_type();
        node.operation_mut().set_result_type(tys.clone());
      }
    }
  }

  println!(
    "       • after transform, egraph size: {}, class size: {}",
    egraph.total_size(),
    egraph.classes().len()
  );

  // 推导bbs信息
  perf_infer::perf_infer(&mut egraph, &vec![root]);

  println!("        • begin vectorize root expression");

  // egraph.dot().to_png("target/foo3.png").unwrap();
  // let mut lp_extractor =
  //   LpExtractor::new(&egraph, ISAXLpCF::new(bb_query.clone()));
  // let expr = lp_extractor.solve(root);

  // for ecls in egraph.classes() {
  //   println!("eclass id: {}, nodes: {:?}", ecls.id, ecls.nodes);
  // }

  let (cost, expr) =
    Extractor::new(&egraph, MaxVectorCF::new(0.9, 1000.0)).find_best(root);
  // let (cost, expr) =
  //   Extractor::new(&egraph, ISAXCF::new(bb_query.clone())).find_best(root);

  // debug!("cost: {:?}", cost);

  let mut vecop_cnt = 0;
  for (id, node) in expr.iter().enumerate() {
    debug!("{}: {:?}", id, node);
    if node.operation().is_vector_op() {
      vecop_cnt += 1;
    }
  }

  let (cycles, area) = rec_cost(&expr, &bb_query, HashMap::new());
  println!("cycles: {}, area: {}", cycles, area);

  println!("        • Vectorized Nodes in root_expr: {}", vecop_cnt);

  for node in expr.clone() {
    if node.operation().get_bbs_info().len() == 0 {
      println!(
        "       [❌]Warning: node {:?} has no bbs info, this may cause issues",
        node
      );
    }
  }
  let mut aus = Vec::new();
  let mut pe_hash = HashSet::new();
  let mut insert_into_aus =
    |pe: PartialExpr<Op, Var>, type_map: HashMap<Var, Vec<String>>| {
      // 检查是否已经存在相同的pe
      if !pe_hash.contains(&pe) {
        pe_hash.insert(pe.clone());
        aus.push((pe, type_map));
      }
    };
  println!("        • Vectorizing root expression...");
  // for (i, node) in expr.iter().enumerate() {
  //   println!("{}: {:?}", i, node);
  // }
  let root_aus = expr_vec2lib(&expr);

  println!(
    "        • vectorize root expression done, size: {}",
    root_aus.len()
  );
  println!("root_aus: {}", root_aus.len());
  for (pe, type_map) in root_aus {
    // println!("pe: {:?}", pe);
    // 将根表达式的au添加到aus中
    insert_into_aus(pe, type_map);
  }

  let mut lib_messages = Vec::new();
  for (i, au) in aus.iter().enumerate() {
    // 如果au的size过小，扔掉
    if au.0.size() < config.liblearn_config.min_lib_size {
      println!(
        "        • Vectorization::lib {}: skip, size: {}",
        i,
        au.0.size()
      );
      continue;
    }

    let searcher: Pattern<_> = au.0.clone().into();
    let results = searcher.search(&egraph);
    println!(
      "        • Vectorization::lib {}: searcher: {}, results: {}",
      i,
      searcher,
      results.len()
    );
    let applier_pe = reify(
      LibId(max_lib_id + i),
      au.0.clone(),
      config.clock_period,
      config.area_estimator.clone(),
      config.delay_estimator.clone(),
      BBQuery::default(),
    );
    let mut applier: Pattern<_> = applier_pe.clone().into();
    // 计算cost
    let ast = &searcher.ast;
    let mut new_expr = ast
      .iter()
      .map(|node| match node {
        egg::ENodeOrVar::ENode(ast_node) => {
          let new_node = (*ast_node).clone();
          new_node
        }
        egg::ENodeOrVar::Var(_) => Op::var(0),
      })
      .collect::<Vec<AstNode<Op>>>();
    // 重新遍历，为var添加bbs信息，方式为遍历每个节点，如果子节点是Var，
    // 则为其添加父节点的bbs信息
    for node in new_expr.clone().iter() {
      for arg in node.args() {
        let op = new_expr[usize::from(*arg)].operation().clone();
        if op.is_var() {
          new_expr[usize::from(*arg)]
            .operation_mut()
            .set_bbs_info(node.operation().get_bbs_info().clone());
        }
      }
    }
    let rec_expr: RecExpr<AstNode<Op>> = new_expr.into();
    let scheduler = Scheduler::new(
      config.clock_period,
      config.area_estimator.clone(),
      config.delay_estimator.clone(),
      bb_query.clone(),
    );
    let (lat_cpu, lat_acc, area) = scheduler.asap_schedule(&rec_expr);
    // 如果area为0，或者lat_acc>=lat_cpu，则continue
    if area == 0 || lat_acc >= lat_cpu {
      println!(
        "        • Vectorization::lib {}: skip, latency cpu: {}, latency acc: {}, area: {}",
        max_lib_id + i,
        lat_cpu,
        lat_acc,
        area
      );
      continue;
    }
    println!(
      "        • Vectorization::lib {}: latency cpu: {}, latency acc: {}, area: {}",
      max_lib_id + i,
      lat_cpu,
      lat_acc,
      area
    );
    // println!("vec_lib{}: {}", i, searcher);
    for node in applier.ast.iter_mut() {
      match node {
        egg::ENodeOrVar::ENode(ast_node) => {
          if let Some(BindingExpr::Lib(id, _, _, _, _, _)) =
            ast_node.as_binding_expr()
          {
            let op = ast_node.operation_mut();
            *op = Op::make_lib(id.into(), lat_cpu, lat_acc, area);
          }
        }
        egg::ENodeOrVar::Var(_) => {}
      };
    }
    let mut condition: HashMap<Var, Vec<T>> = HashMap::new();
    for (var, tys) in au.1.iter() {
      let tys_parse = tys
        .iter()
        .map(|ty| ty.parse::<T>().unwrap_or_else(|_| unreachable!()))
        .collect::<Vec<_>>();
      condition.insert(var.clone(), tys_parse);
    }
    let condition = TypeMatch::new(condition);
    // 用searcher搜索egraph，取得results.len()
    let results = searcher.search(&egraph);
    lib_messages.push(LiblearnMessage {
      lib_id: max_lib_id + i,
      rewrite: Rewrite::new(
        format!("vec_lib_{}", i),
        searcher.clone(),
        applier.clone(),
      )
      .unwrap_or_else(|_| unreachable!()),
      searcher_pe: au.0.clone(),
      applier_pe: applier_pe,
      condition: condition,
      search_len: results.len(),
    });
  }
  println!(
    "        • Vectorize:: There are {} vec_libs",
    lib_messages.len()
  );
  for message in lib_messages.iter() {
    debug!(
      "vec_lib: {}, results: {}",
      Pattern::from(message.applier_pe.clone()),
      message.search_len
    );
  }

  // 建立fused EGraph,在egraph_without_dsrs的基础上，添加向量化的expr,
  // 之后将两个id union起来
  let add_vec_expr2_egraph = |egraph: &EGraph<
    AstNode<Op>,
    ISAXAnalysis<Op, T>,
  >|
   -> (
    EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
    Id,
  ) {
    let mut new_egraph = egraph.clone();
    // 将expr添加到new_egraph中
    let new_id = new_egraph.add_expr(&expr);
    new_egraph.union(new_id, root);
    new_egraph.rebuild();
    // 之后加入get节点
    add_get_to_build_attach(&mut new_egraph, lower_dsrs);
    let egraph = new_egraph.clone();
    (egraph.clone(), egraph.find(new_id))
  };

  let new_egraph_running_scalar_dsrs =
    add_vec_expr2_egraph(&egraph_running_scalar_dsrs);

  println!(
    "       • after vectorization, egraph(running scalar dsrs) size: {},
  class size: {}",
    new_egraph_running_scalar_dsrs.0.total_size(),
    new_egraph_running_scalar_dsrs.0.classes().len()
  );

  (new_egraph_running_scalar_dsrs, lib_messages)
}

fn can_add_get<Op, T>(
  egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  enode: &AstNode<Op>,
) -> bool
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + BBInfo
    + Arity
    + Send
    + Sync
    + Debug
    + 'static
    + DiscriminantEq
    + Default
    + Printable
    + OperationInfo,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + FromStr
    + 'static,
  AstNode<Op>: TypeInfo<T>,
{
  if enode.operation().is_vec() {
    // 如果是vec节点，直接返回false
    return false;
  }
  let is_lit_vec = |n: &AstNode<Op>| {
    // 检查是否是vec节点，并且args全为0
    if n.operation().is_vec() {
      let args = n.args();
      return args
        .iter()
        .all(|&a| egraph[a].nodes.iter().all(|n| n.args().len() == 0));
    }
    false
  };
  if enode.operation().is_vector_op() {
    // 如果是vector_op节点，需要其子节点不是vec节点
    for arg in enode.args() {
      if egraph[*arg]
        .nodes
        .iter()
        .any(|n| n.operation().is_vec() && !is_lit_vec(n))
      {
        return false;
      }
    }
    return true;
  }
  false
}

pub fn add_get_to_build_attach<Op, T>(
  egraph: &mut EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  lower_dsrs: &Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
) where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + BBInfo
    + Arity
    + Send
    + Sync
    + Debug
    + 'static
    + DiscriminantEq
    + Default
    + Printable
    + OperationInfo,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + FromStr
    + 'static,
  AstNode<Op>: TypeInfo<T>,
{
  // println!("egraph.classes().len(): {}", egraph.classes().len());
  // 此处是为了避免添加循环
  // let can_add_get = |enode: &AstNode<Op>| -> bool {
  //   if enode.operation().is_vec() {
  //     // 如果是vec节点，直接返回false
  //     return false;
  //   }
  //   if enode.operation().is_vector_op() {
  //     // 如果是vector_op节点，需要其子节点不是vec节点
  //     for arg in enode.args() {
  //       if egraph[*arg].nodes.iter().any(|n| !can_add_get(n)) {
  //         return false;
  //       }
  //     }
  //     return true;
  //   }
  //   false
  // };
  let is_lit_vec = |n: &AstNode<Op>| {
    // 检查是否是vec节点，并且args全为0
    if n.operation().is_vec() {
      let args = n.args();
      return args
        .iter()
        .all(|&a| egraph[a].nodes.iter().all(|n| n.args().len() == 0));
    }
    false
  };
  // for ecls in egraph.classes() {
  //   println!("original eclass: {}, nodes: {:?}", ecls.id, ecls.nodes);
  // }
  let mut vec_op_nodes = Vec::new();
  let mut vec_nodes = Vec::new();
  for class in egraph.classes() {
    for node in class.nodes.iter() {
      if can_add_get(egraph, node) {
        vec_op_nodes.push((class.id, node.operation().get_vec_len()));
      } else if node.operation().is_vector_op() && !is_lit_vec(node) {
        // 如果是vec节点，记录下来
        vec_nodes.push((class.id, node.operation().get_vec_len()));
      }
    }
  }
  // println!("vec_op_nodes: {:?}", vec_op_nodes);
  // println!("vec_nodes: {:?}", vec_nodes);
  // 对于每个vector_op节点，添加get节点
  for (id, vec_len) in vec_op_nodes {
    // 对于每个vec_len，添加get节点
    let bbs = egraph[id].data.bb.clone();
    for i in 0..vec_len {
      let get_node = AstNode::new(
        Op::make_get_from_vec(i, vec!["unknown".to_string()], bbs.clone()),
        vec![id],
      );
      let _ = egraph.add(get_node);
    }
  }
  // 重新构建egraph
  egraph.rebuild();

  println!(
    "       • after add get, egraph size: {}, class size: {}",
    egraph.total_size(),
    egraph.classes().len()
  );

  let origin_ids: Vec<Id> = egraph.classes().map(|class| class.id).collect();

  let mut mirror_egraph = egraph.clone();
  // 为刚才没有加入的vec节点添加get节点
  for (id, vec_len) in vec_nodes {
    // 对于每个vec_len，添加get节点
    let bbs = egraph[id].data.bb.clone();
    for i in 0..vec_len {
      let get_node = AstNode::new(
        Op::make_get_from_vec(i, vec!["unknown".to_string()], bbs.clone()),
        vec![id],
      );
      let _ = mirror_egraph.add(get_node);
    }
  }

  // 用一个mirror runner运行lower_dsrs
  let runner = Runner::<_, _, ()>::new(ISAXAnalysis::empty())
    .with_egraph(mirror_egraph)
    .with_time_limit(Duration::from_secs(60 * 100_000))
    .with_iter_limit(1000)
    .run(lower_dsrs);
  // for ecls in runner.egraph.classes() {
  //   println!("lower eclass: {}, nodes: {:?}", ecls.id, ecls.nodes);
  // }
  let mut new_union_map = HashMap::new();
  for id in origin_ids {
    let new_id = runner.egraph.find(id);
    new_union_map
      .entry(new_id)
      .or_insert_with(Vec::new)
      .push(id);
  }
  for (_, old_ids) in new_union_map {
    if old_ids.len() > 1 {
      // 如果有多个old_id对应同一个new_id，则进行union
      let first_id = old_ids[0];
      for &id in &old_ids[1..] {
        egraph.union(egraph.find(first_id), egraph.find(id));
      }
    }
  }

  // 重新构建egraph
  egraph.rebuild();
  // egraph.dot().to_png("target/foo4.png").unwrap();
  // for ecls in egraph.classes() {
  //   println!("eclass: {}, nodes: {:?}", ecls.id, ecls.nodes);
  // }
  println!(
    "       • after running lower dsrs, egraph size: {}, class size: {}",
    egraph.total_size(),
    egraph.classes().len()
  );
}

// pub struct VecExtractor<'a, Op, T>
// where
//   Op: Display
//     + Hash
//     + Clone
//     + Ord
//     + Teachable
//     + BBInfo
//     + Arity
//     + Send
//     + Sync
//     + Debug
//     + OperationInfo
//     + 'static,
//   T: Debug + Default + Clone + PartialEq + Ord + Hash + Send + Sync,
//   AstNode<Op>: TypeInfo<T>,
// {
//   cost_function: VectorCF,
//   costs: HashMap<Id, (f64, AstNode<Op>)>,
//   costs_get_vec: HashMap<Id, (f64, AstNode<Op>)>,
//   egraph: &'a EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
// }

// impl<'a, Op, T> VecExtractor<'a, Op, T>
// where
//   Op: Display
//     + Hash
//     + Clone
//     + Ord
//     + Teachable
//     + BBInfo
//     + Arity
//     + Send
//     + Sync
//     + Debug
//     + OperationInfo
//     + 'static,
//   T: Debug + Default + Clone + PartialEq + Ord + Hash + Send + Sync,
//   AstNode<Op>: TypeInfo<T>,
// {
//   /// Create a new `VecExtractor` given an `EGraph`.
//   pub fn new(
//     egraph: &'a EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
//     cost_function: VectorCF,
//   ) -> Self {
//     let mut extractor = VecExtractor {
//       costs: HashMap::default(),
//       costs_get_vec: HashMap::default(),
//       egraph,
//       cost_function,
//     };
//     extractor.find_costs();
//     // 打印costs和 costs_get_vec
//     // for (id, (cost, node)) in extractor.costs.iter() {
//     //   println!("Cost for eclass {}: {:?} = {}", id, node, cost);
//     // }
//     // for (id, (cost, node)) in extractor.costs_get_vec.iter() {
//     //   println!("GetVec Cost for eclass {}: {:?} = {}", id, node, cost);
//     // }
//     extractor
//   }

//   /// Find the cheapest (lowest cost) represented `RecExpr` in the
//   /// given eclass.
//   pub fn find_best(&self, eclass: Id) -> (f64, RecExpr<AstNode<Op>>) {
//     // println!("Finding best for eclass: {}", &self.egraph.find(eclass));
//     let (cost, root) = self.costs[&self.egraph.find(eclass)].clone();
//     // println!("Best cost: {}", cost);
//     let mut visited = HashSet::new();
//     let expr = self.build_recexpr_from_node(&root, &mut visited);
//     (cost, expr)
//   }

//   /// Find the cheapest e-node in the given e-class.
//   pub fn find_best_normal_node(&self, eclass: Id) -> Option<&AstNode<Op>> {
//     if !self.costs.contains_key(&self.egraph.find(eclass)) {
//       return None; // 如果没有找到对应的e-class，返回None
//     }
//     let node = &self.costs[&self.egraph.find(eclass)].1;
//     Some(node)
//   }

//   /// Find the cheapest e-node in the given e-class that is a `get` operation
//   /// and has a vector operation as its child.
//   pub fn find_best_get_vec_node(&self, eclass: Id) -> Option<&AstNode<Op>> {
//     if !self.costs_get_vec.contains_key(&self.egraph.find(eclass)) {
//       return None; // 如果没有找到对应的e-class，返回None
//     }
//     let node = &self.costs_get_vec[&self.egraph.find(eclass)].1;
//     Some(node)
//   }

//   /// Find the cost of the term that would be extracted from this e-class.
//   pub fn find_best_cost(&self, eclass: Id) -> f64 {
//     let (cost, _) = &self.costs[&self.egraph.find(eclass)];
//     cost.clone()
//   }

//   fn node_total_cost(&mut self, node: &AstNode<Op>) -> Option<f64> {
//     let eg = &self.egraph;
//     let has_cost = |id| self.costs.contains_key(&eg.find(id));
//     if node.all(has_cost) {
//       let costs = &self.costs;
//       let cost_f = |id| costs[&eg.find(id)].0.clone();
//       Some(self.cost_function.cost(node, cost_f))
//     } else {
//       None
//     }
//   }

//   fn find_costs(&mut self) {
//     let mut did_something = true;
//     while did_something {
//       did_something = false;

//       for class in self.egraph.classes() {
//         let (normal_pass, vec_pass) = self.make_pass(class);
//         match (self.costs.get(&class.id), normal_pass) {
//           (None, Some(new)) => {
//             // println!(
//             //   "Adding cost for eclass {}: {:?} = {:?}",
//             //   class.id, new.1, new.0
//             // );
//             self.costs.insert(class.id, new);
//             did_something = true;
//           }
//           (Some(old), Some(new)) => {
//             if new.0 < old.0 {
//               // println!(
//               //   "Updating cost for eclass {}:  from {:?} to {:?}",
//               //   class.id, old.1, new.1
//               // );
//               self.costs.insert(class.id, new);
//               did_something = true;
//             } else {
//               // println!(
//               //   "Skipping eclass {}: {:?} (cost not improved)",
//               //   class.id, new.1
//               // );
//             }
//           }
//           _ => {
//             // println!("Skipping eclass {}: {:?} ", class.id, class.nodes)
//           }
//         }
//         match (self.costs_get_vec.get(&class.id), vec_pass) {
//           (None, Some(new)) => {
//             self.costs_get_vec.insert(class.id, new);
//             did_something = true;
//           }
//           (Some(old), Some(new)) => {
//             if new.0 < old.0 {
//               self.costs_get_vec.insert(class.id, new);
//               did_something = true;
//             }
//           }
//           _ => {
//             // println!("Skipping eclass {}: {:?} ", class.id, class.nodes)
//           }
//         }
//       }
//     }

//     for class in self.egraph.classes() {
//       if !self.costs.contains_key(&class.id) {
//         log::warn!(
//           "Failed to compute cost for eclass {}: {:?}",
//           class.id,
//           class.nodes
//         )
//       }
//     }
//   }

//   fn make_pass(
//     &mut self,
//     eclass: &EClass<AstNode<Op>, ISAXCost<T>>,
//   ) -> (Option<(f64, AstNode<Op>)>, Option<(f64, AstNode<Op>)>) {
//     // println!("Making pass for eclass: {}", eclass.id);
//     // 先不用 clone 整个 Vec，若 EClass 支持.iter()，应优先 iter
//     let mut get_vec_nodes = Vec::new();
//     let mut other_nodes = Vec::new();
//     for node in eclass.iter().cloned() {
//       // 确保 args 至少有一个
//       if node.operation().is_get_from_vec() && node.args().len() >= 1 {
//         let child_id = node.args()[0];
//         // self.egraph[child_id] 取子 e-class，再检查其中是否含 vec op
//         if self.egraph[child_id].iter().any(|n| n.operation().is_vec()) {
//           get_vec_nodes.push(node);
//           continue;
//         }
//       }
//       other_nodes.push(node);
//     }

//     // helper: 选最小成本节点
//     let select_best = |nodes: &[AstNode<Op>],
//                        extractor: &mut Self|
//      -> Option<(f64, AstNode<Op>)> {
//       // 过滤出 node_total_cost 有 Some 的
//       let mut best: Option<(f64, AstNode<Op>)> = None;
//       for n in nodes.iter() {
//         if let Some(c) = extractor.node_total_cost(n) {
//           if c.is_nan() {
//             continue; // 或者 treat as None
//           }
//           match &best {
//             None => best = Some((c, n.clone())),
//             Some((best_c, _)) => {
//               // 小于才替换
//               if c < *best_c {
//                 best = Some((c, n.clone()));
//               }
//             }
//           }
//         }
//         // println!("Node {:?} has cost: {:?}", n,
//         // extractor.node_total_cost(n));
//       }
//       best
//     };

//     // 先看 get_vec_nodes
//     let vec_pass = if !get_vec_nodes.is_empty() {
//       if let Some(best) = select_best(&get_vec_nodes, self) {
//         // println!("选中了 get->vec 节点：cost={}, node={:?}", best.0,
// best.1);         Some(best)
//       } else {
//         None
//       }
//     } else {
//       None
//     };

//     // 再选 other_nodes
//     let normal_pass = if !other_nodes.is_empty() {
//       if let Some(best) = select_best(&other_nodes, self) {
//         // println!("选中了普通节点：cost={}, node={:?}", best.0, best.1);
//         Some(best)
//       } else {
//         None
//       }
//     } else {
//       None
//     };
//     (normal_pass, vec_pass)
//   }

//   /// 给定一个根 AstNode<Op>（通常是 costs 中存储的最佳节点），以及
//   /// 一个回调 get_node: FnMut(Id) -> AstNode<Op>（负责返回子 eclass
//   /// 对应最佳节点）， 构建 RecExpr<AstNode<Op>>。此函数不会返回 Err，因为
//   /// get_node 返回非 Result。
//   fn build_recexpr_from_node(
//     &self,
//     root: &AstNode<Op>,
//     visited: &mut HashSet<Id>,
//   ) -> RecExpr<AstNode<Op>> {
//     // 把 get_node 包装成一个总是 Ok 的版本，再调用 try 版
//     self
//       .try_build_recexpr_from_node::<Infallible>(root, visited)
//       .unwrap()
//   }

//   /// 可 fallible 版本：get_node 可能返回 Err，则整个构建可以失败。
//   fn try_build_recexpr_from_node<Err>(
//     &self,
//     root: &AstNode<Op>,
//     visited: &mut HashSet<Id>,
//   ) -> Result<RecExpr<AstNode<Op>>, Err> {
//     let get_node = |id: Id| -> Option<&AstNode<Op>> {
//       // 直接从 costs 中获取最佳节点
//       self.find_best_normal_node(id).clone()
//     };
//     let get_vec_node = |id: Id| -> Option<&AstNode<Op>> {
//       // 直接从 costs_get_vec 中获取最佳 get_vec 节点
//       self.find_best_get_vec_node(id).clone()
//     };
//     // IndexSet 用于去重并保持插入顺序
//     let mut set = IndexSet::<AstNode<Op>>::default();
//     // ids: 原 eclass Id -> RecExpr 中对应的新 Id
//     let mut ids = HashMap::<Id, Id>::default();
//     // todo 栈：初始化为根节点的子节点列表
//     let mut todo = root
//       .children()
//       .iter()
//       .map(|&id| {
//         let get_node = get_vec_node(id);
//         if get_node.is_some() {
//           // 如果是 get_vec 节点，直接使用它
//           visited.insert(id);
//           (id, true)
//         } else {
//           // 否则使用普通节点
//           (id, false)
//         }
//       })
//       .collect::<Vec<_>>();

//     // 处理栈，直到空
//     while let Some((id, is_get_vec)) = todo.last().copied() {
//       // 如果已处理过，即 ids 中存在，则跳过
//       if ids.contains_key(&id) {
//         todo.pop();
//         continue;
//       }
//       // 通过 get_node 回调获得 AstNode<Op>，若 Err 则返回 Err
//       let node = if is_get_vec {
//         get_vec_node(id)
//       } else {
//         get_node(id)
//       };

//       if node.is_none() {
//         // 如果找不到节点，返回 Err
//         panic!(
//           "Failed to find node for id {} in eclass {}",
//           id,
//           self.egraph.find(id)
//         );
//       }
//       let node = node.unwrap();

//       // 检查 node 的所有子 id 是否都已在 ids 中
//       let mut all_ready = true;
//       for &child in node.children() {
//         if !ids.contains_key(&child) {
//           all_ready = false;
//           // 如果有子节点未就绪，则将其 push 到 todo 栈中
//           //
// 此时需要检查，如果没有visit过，并且get_vec_node有值，那么就插入true
//           if !visited.contains(&child) {
//             let is_get_vec = get_vec_node(child).is_some();
//             todo.push((child, is_get_vec));
//             visited.insert(child);
//           } else {
//             todo.push((child, false));
//           }
//         }
//       }
//       // 如果所有子已就绪，则可以安全插入
//       if all_ready {
//         // 将子节点引用替换为已在 RecExpr 中的 Id
//         let mapped = node.clone().map_children(|child_id| ids[&child_id]);
//         // 插入 set，获取新插入位置 index
//         let new_index = set.insert_full(mapped).0;
//         // 记录原 eclass id -> RecExpr Id
//         ids.insert(id, Id::from(new_index));
//         // 弹出栈顶
//         todo.pop();
//       }
//       // 否则已把未就绪的子 push 入 todo，下一轮先处理子
//     }

//     // 构造 RecExpr：先把 set 中的所有节点按顺序收集
//     let mut expr: RecExpr<AstNode<Op>> = set.into_iter().collect();
//     // 最后插入根节点自身，map_children 映射子 id
//     let root_mapped = root.clone().map_children(|child_id| {
//       // child_id 必定在 ids 中
//       ids[&child_id]
//     });
//     expr.add(root_mapped);
//     Ok(expr)
//   }
// }

// 构建模板的递归辅助函数
fn build_template<
  Op: OperationInfo + Ord + Debug + Clone + Arity + Teachable + Hash,
>(
  rec_expr: &RecExpr<AstNode<Op, Id>>,
  root: Id,
  cache: &mut HashMap<Id, PartialExpr<Op, Var>>,
  var_index: &mut usize,
  type_map: &mut HashMap<Var, Vec<String>>,
) -> PartialExpr<Op, Var> {
  if let Some(cached) = cache.get(&root) {
    return cached.clone();
  }

  let node = &rec_expr[root];
  let op = node.operation();

  // 检查是否需要替换为Hole
  if !op.is_vector_op() || op.is_vec() {
    *var_index += 1;
    let var_name = format!("?x{}", *var_index - 1);
    let var: Var = var_name.parse().unwrap_or_else(|_| unreachable!());
    type_map.insert(var.clone(), op.get_result_type());
    let hole = PartialExpr::Hole(var);
    cache.insert(root, hole.clone());
    hole
  } else {
    // 处理子节点
    let mut new_children = Vec::new();
    for child_id in node.children() {
      let child_expr =
        build_template(rec_expr, *child_id, cache, var_index, type_map);
      new_children.push(child_expr);
    }

    // 创建新节点
    let new_node = PartialExpr::Node(AstNode::new(op.clone(), new_children));
    cache.insert(root, new_node.clone());
    new_node
  }
}

// 主转换函数
fn expr_vec2lib<
  Op: OperationInfo + Ord + Debug + Clone + Arity + Teachable + Hash,
>(
  rec_expr: &RecExpr<AstNode<Op, Id>>,
) -> Vec<(PartialExpr<Op, Var>, HashMap<Var, Vec<String>>)> {
  if rec_expr.is_empty() {
    return Vec::new();
  }

  let n = rec_expr.len();

  // —— 1. 构建父节点列表 ——
  let mut parents: Vec<Vec<usize>> = vec![Vec::new(); n];
  for i in 0..n {
    for &child in rec_expr[Id::from(i)].children() {
      parents[usize::from(child)].push(i);
    }
  }

  // —— 2. 筛选出所有「符合条件」的 vector_op 节点 ——
  let mut libs = Vec::new();
  for i in 0..n {
    let node = &rec_expr[Id::from(i)];
    let op = node.operation();
    // 条件①：本身是 vector_op 且不是 Vec 构造
    if !op.is_vector_op() || op.is_vec() {
      continue;
    }
    // 条件②：所有父节点都是 Vec 构造 or 不是 vector_op（即 scalar）
    if !parents[i].iter().all(|&pi| {
      let pop = rec_expr[Id::from(pi)].operation();
      pop.is_vec() || !pop.is_vector_op()
    }) {
      continue;
    }

    // 满足条件，提取这个子表达式模板
    let mut local_cache = HashMap::new();
    let mut var_index = 0;
    let mut type_map = HashMap::new();
    let body = build_template(
      rec_expr,
      Id::from(i),
      &mut local_cache,
      &mut var_index,
      &mut type_map,
    );
    libs.push((body, type_map));
  }

  libs
}
