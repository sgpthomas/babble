//! Runner module for executing library learning experiments
//!
//! This module provides functionality for running library learning experiments
//! using either regular beam search or Pareto-optimal beam search.

use std::{
  collections::{HashMap, HashSet},
  fmt::{Debug, Display},
  hash::{DefaultHasher, Hash, Hasher},
  str::FromStr,
  time::{Duration, Instant},
};

use bitvec::{bitvec, order::Lsb0, vec};
use lexpr::print;
use nom::lib;
use serde::Deserialize;

use crate::{
  analysis::SimpleAnalysis,
  au_filter::{CiEncodingConfig, TypeAnalysis},
  bb_query::{self, BBInfo, BBQuery},
  expand::{expand, ExpandMessage, MetaAUConfig},
  extract::beam_pareto::{
    compute_full_hash, compute_hash_level, eliminate_lambda, ClassMatch,
    ISAXAnalysis, ISAXCost, ISAXLpCF, LevelConflictState, LibExtractor,
    StructuralHash, TypeInfo, TypeSet,
  },
  perf_infer,
  rewrites::{self, TypeMatch},
  schedule::{
    cycles_for_every_function, dump_rec_cost, rec_cost, Schedulable, Scheduler,
  },
  vectorize::{vectorize, VectorConfig},
  Arity, AstNode, BindingExpr, DiscriminantEq, Expr, LearnedLibraryBuilder, LibId,
  PartialExpr, Pretty, Printable, Teachable,
};
use egg::{
  Analysis, EGraph, Extractor, Id, Language, LpExtractor, Pattern, RecExpr,
  Rewrite, Runner as EggRunner, Searcher, SimpleScheduler, Var,
};
use log::{debug, info};

pub trait OperationInfo {
  // 拿到全部信息
  fn get_full_info(&self) -> String;
  // 是否满足交换律
  fn is_commutative(&self) -> bool;
  fn is_dowhile(&self) -> bool;
  fn is_liblearn_banned_op(&self) -> bool;
  fn is_lib(&self) -> bool;
  fn is_list_op(&self) -> bool;
  fn is_lib_op(&self) -> bool;
  /// Get the library ID of the operation
  fn get_libid(&self) -> usize;
  /// Make a lib node
  fn make_lib(id: LibId, lat_cpu: f64, lat_acc: f64, cost: usize) -> Self;
  /// make a list op
  fn make_vec(result_tys: Vec<String>, bbs: Vec<String>) -> Self;
  /// make a get_from_vec op
  fn make_get_from_vec(
    id: usize,
    result_tys: Vec<String>,
    bbs: Vec<String>,
  ) -> Self;
  /// Get the constant value of the operation
  fn get_const(&self) -> Option<(i64, u32)>;
  /// Make an AST node for type-awared Const
  fn make_const(const_value: (i64, u32)) -> Self;
  /// judge whether a node is dummy
  fn is_dummy(&self) -> bool;
  /// judge whether a node is a vector op
  fn is_vector_op(&self) -> bool;
  /// get_simple_cost
  fn get_simple_cost(&self) -> f64;
  /// For List and Vec, though args may have different order , but they are the
  /// sam, this function is used to judge
  fn is_vec(&self) -> bool {
    false
  }
  /// make gather node
  fn make_gather(indices: &Vec<usize>, bbs: Vec<String>) -> Self;
  /// make shuffle node
  fn make_shuffle(indices: &Vec<usize>, bbs: Vec<String>) -> Self;
  /// get_bbs_info: will be used in vectorization
  fn get_bbs_info(&self) -> Vec<String> {
    vec![]
  }
  /// 获得当前操作符的类型
  fn get_result_type(&self) -> Vec<String> {
    vec![]
  }
  /// 为每个Operation设置返回类型
  fn set_result_type(&mut self, result_ty: Vec<String>);
  /// get_vec_len
  fn get_vec_len(&self) -> usize {
    1
  }
  /// 加入Op_pack节点
  fn make_op_pack(ops: Vec<String>, bbs: Vec<String>) -> Self;
  /// Group a packed operation for Meta AU resource-sharing diagnostics.
  fn op_pack_area_family(op: &str) -> &'static str {
    let _ = op;
    "other"
  }
  /// 加入Op_select节点
  fn make_op_select(idx: usize) -> Self;
  /// 加入rule_var节点
  fn make_rule_var(name: String) -> Self;
  /// 加入Opmask节点
  fn make_opmask() -> Self;
  /// 是不是Opmask节点
  fn is_opmask(&self) -> bool {
    false
  }
  /// 是不是RuleVar节点
  fn is_rule_var(&self) -> bool {
    false
  }
  fn get_bitwidth(&self) -> usize;
  fn make_bitwidth(&mut self, width: usize);
  /// 是不是var节点
  fn is_var(&self) -> bool {
    false
  }
  /// 设置bbs信息
  fn set_bbs_info(&mut self, bbs: Vec<String>);
  fn is_arithmetic(&self) -> bool;
  fn is_op(&self) -> bool;
  /// 是不是tuple节点
  fn is_tuple(&self) -> bool {
    false
  }
  /// 是不是get_from_arg节点
  fn is_get_from_arg(&self) -> bool {
    false
  }
  /// 是不是get_from_vec节点
  fn is_get_from_vec(&self) -> bool {
    false
  }
  /// 是不是external_arg
  fn is_external_arg(&self) -> bool {
    false
  }
  /// 通过节点op和其子节点的Op的类型，判断这个节点是不是有用的
  fn is_useful_expr(&self, children_ops: &[Self]) -> bool
  where
    Self: Sized,
  {
    // 默认返回true
    true
  }
  fn is_mem(&self) -> bool;
  /// Whether two operations may be generalized through Meta AU.
  ///
  /// Implementations should keep this conservative: Meta AU abstracts the
  /// operator itself, so effectful operations or operations from different
  /// semantic families should not be packed together.
  fn meta_au_compatible_with(&self, other: &Self) -> bool
  where
    Self: Sized,
  {
    self.is_arithmetic()
      && other.is_arithmetic()
      && !self.is_mem()
      && !other.is_mem()
  }
  /// Returns a coarser operator identity for Meta AU structural hashing.
  ///
  /// The default keeps the original operator identity by returning `None`.
  /// Meta AU implementations may override this to normalize compatible
  /// operators into a shared family-level key so that pair pruning does not
  /// accidentally discard cross-operator matches such as `add/sub`.
  fn meta_au_hash_key(&self) -> Option<String> {
    None
  }
  fn op_execution_count(&self, bb_query: &BBQuery) -> f64;
  // 在lib_learn的时候，有时候需要删除某些信息，将操作符generic化
  fn genericize(&mut self);
}

/// A trait for running library learning experiments with Pareto optimization
pub trait BabbleParetoRunner<Op, T>
where
  Op: Arity
    + Teachable
    + Printable
    + Debug
    + Display
    + Hash
    + Clone
    + Ord
    + Sync
    + Send
    + DiscriminantEq
    + 'static,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + 'static
    + Send
    + Sync
    + Display,
  AstNode<Op>: TypeInfo<T>,
{
  /// Run the experiment on a single expression
  fn run(&self, expr: Expr<Op>) -> ParetoResult<Op, T>;

  /// Run the experiment on multiple expressions
  fn run_multi(&self, exprs: Vec<Expr<Op>>) -> ParetoResult<Op, T>;

  /// Run the experiment on groups of equivalent expressions
  fn run_equiv_groups(
    &self,
    expr_groups: Vec<Vec<Expr<Op>>>,
  ) -> ParetoResult<Op, T>;
}

/// A Egraph with (root, latency, area) information
#[derive(Clone, Debug)]
pub struct ExtractResult<Op, T>
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Debug
    + 'static
    + OperationInfo
    + Send
    + Sync,
  T: Debug + Ord + Clone + Default + Hash + 'static + Send + Sync,
  AstNode<Op>: TypeInfo<T>,
{
  pub expr: RecExpr<AstNode<Op>>,
  pub libs: HashMap<usize, Pattern<AstNode<Op>>>,
  pub cycles: f64,
  pub area: usize,
  pub func_cycles: HashMap<String, f64>,
  pub lib_reuses: HashMap<usize, usize>,
  _phantom: std::marker::PhantomData<T>,
}

impl<Op, T> PartialEq for ExtractResult<Op, T>
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Debug
    + 'static
    + OperationInfo
    + Send
    + Sync,
  T: Debug + Ord + Clone + Default + Hash + 'static + Send + Sync,
  AstNode<Op>: TypeInfo<T>,
{
  fn eq(&self, other: &Self) -> bool {
    self.expr == other.expr
      && self.cycles == other.cycles
      && self.area == other.area
  }
}

impl<Op, T> ExtractResult<Op, T>
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Debug
    + 'static
    + OperationInfo
    + Send
    + Sync,
  T: Debug + Ord + Clone + Default + Hash + 'static + Send + Sync,
  AstNode<Op>: TypeInfo<T>,
{
  pub fn new(
    expr: RecExpr<AstNode<Op>>,
    libs: HashMap<usize, Pattern<AstNode<Op>>>,
    cycles: f64,
    area: usize,
    func_cycles: HashMap<String, f64>,
    lib_reuses: HashMap<usize, usize>,
  ) -> Self {
    Self {
      expr,
      libs,
      cycles,
      area,
      func_cycles,
      lib_reuses,
      _phantom: std::marker::PhantomData,
    }
  }
}

/// Result of running a BabbleRunner experiment
#[derive(Clone)]
pub struct ParetoResult<Op, T>
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Debug
    + 'static
    + OperationInfo
    + Send
    + Sync,
  T: Debug + Ord + Clone + Default + Hash + 'static + Send + Sync + Display,
  AstNode<Op>: TypeInfo<T>,
{
  /// The number of libraries learned
  pub num_libs: usize,
  /// The final result of pareto: a series of AnnotatedEGraph
  pub extract_results: Vec<ExtractResult<Op, T>>,
  /// when vectorizing, vectorized_expr is needed
  pub vectorized_egraph_with_root:
    Option<(EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>, Id)>,
  /// the message for the chosen libs
  pub lib_message: ExpandMessage<Op, T>,
  /// The time taken to run the experiment
  pub run_time: Duration,
  /// message map to record the message
  pub egraph_size: (usize, usize),
}

impl<Op, T> std::fmt::Debug for ParetoResult<Op, T>
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Debug
    + 'static
    + OperationInfo
    + Send
    + Sync,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + 'static
    + Send
    + Sync
    + Display,
  AstNode<Op>: TypeInfo<T>,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("ParetoResult")
      .field("num_libs", &self.num_libs)
      // .field("perf_gain", &self.perf_gain)
      // .field("area", &self.area)
      .field("run_time", &self.run_time)
      .finish()
  }
}

/// Configuration for beam search
#[derive(Debug, Clone)]
pub struct ParetoConfig<LA, LD>
where
  LA: Debug + Default + Clone,
  LD: Debug + Default + Clone,
{
  /// The final beam size to use
  pub final_beams: usize,
  /// The inter beam size to use
  pub inter_beams: usize,
  /// The number of times to apply library rewrites
  pub lib_iter_limit: usize,
  /// The number of libs to learn at a time
  pub lps: usize,
  /// Whether to learn "library functions" with no arguments
  pub learn_constants: bool,
  /// Maximum arity of a library function
  pub max_arity: Option<usize>,
  /// clock period used for scheduling
  pub clock_period: usize,
  /// area estimator
  pub area_estimator: LA,
  /// delay estimator
  pub delay_estimator: LD,
  /// whether to add all types
  pub enable_widths_merge: bool,
  /// liblearn config
  pub liblearn_config: LiblearnConfig,
  /// vectorize config
  pub vectorize_config: VectorConfig,
  /// op_pack config
  pub op_pack_config: MetaAUConfig,
  /// ci_encoding config
  pub ci_encoding_config: CiEncodingConfig,
  /// find_pack config
  pub find_pack_config: FindPackConfig,
}

impl<LA, LD> Default for ParetoConfig<LA, LD>
where
  LA: Debug + Default + Clone,
  LD: Debug + Default + Clone,
{
  fn default() -> Self {
    Self {
      final_beams: 10,
      inter_beams: 5,
      lib_iter_limit: 3,
      lps: 1,
      learn_constants: false,
      max_arity: None,
      clock_period: 1000,
      area_estimator: LA::default(),
      delay_estimator: LD::default(),
      enable_widths_merge: false,
      liblearn_config: LiblearnConfig::default(),
      vectorize_config: VectorConfig::default(),
      op_pack_config: MetaAUConfig::default(),
      ci_encoding_config: CiEncodingConfig::default(),
      find_pack_config: FindPackConfig::default(),
    }
  }
}

/// Config for Learning Library
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LiblearnCost {
  /// type of cost : "delay", "size"， "latencygainarea"
  Delay,
  Size,
  LatencyGainArea,
}
impl Default for LiblearnCost {
  fn default() -> Self {
    Self::Delay
  }
}
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AUMergeMod {
  /// type of AU merge : "random", "kd", "boundary", "cartesian"
  Random,
  Kd,
  Boundary,
  #[serde(rename = "cartesian")]
  Cartesian,
}
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EnumMode {
  /// enumerate mode: "all", "pruning vanilla", "pruning gold", "cluster test"
  All,
  PruningVanilla,
  PruningGold,
  ClusterTest,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub struct LiblearnConfig {
  /// type of cost : "delay", "Match", "size"
  pub cost: LiblearnCost,
  /// type of AU merge : "random", "kd", "greedy"
  pub au_merge_mod: AUMergeMod,
  /// enumerate mode: "all", "pruning vanilla", "pruning gold"
  pub enum_mode: EnumMode,
  /// for greedy au merge , we only merge the biggest AU, but the other two
  /// need to sample m AUs
  pub sample_num: usize,
  /// to judge whether two eclasses are similar, we need to use two thresholds:
  /// Hamming distance and Jaccard distance
  pub hamming_threshold: usize,
  pub jaccard_threshold: f64,
  /// every liblearn, we learn at most this number of libs
  pub max_libs: usize,
  /// a lib must have at least this number of nodes
  pub min_lib_size: usize,
  /// a lib can have at most this number of nodes
  pub max_lib_size: usize,
}

impl Default for LiblearnConfig {
  fn default() -> Self {
    Self {
      cost: LiblearnCost::Delay,
      au_merge_mod: AUMergeMod::Boundary,
      enum_mode: EnumMode::All,
      sample_num: 10,
      hamming_threshold: 36,
      jaccard_threshold: 0.67,
      max_libs: 500,
      min_lib_size: 4,
      max_lib_size: 500,
    }
  }
}

impl LiblearnConfig {
  /// Create a new LiblearnConfig with the given parameters
  pub fn new(
    cost: LiblearnCost,
    au_merge_mod: AUMergeMod,
    enum_mode: EnumMode,
    sample_num: usize,
    hamming_threshold: usize,
    jaccard_threshold: f64,
    max_libs: usize,
    min_lib_size: usize,
    max_lib_size: usize,
  ) -> Self {
    Self {
      cost,
      au_merge_mod,
      enum_mode,
      sample_num,
      hamming_threshold,
      jaccard_threshold,
      max_libs,
      min_lib_size,
      max_lib_size,
    }
  }
}

/// FindPackConfig用于在向量化和MetaAU的过程中使用，会对lib-learn做出一些放宽，
/// 这些放宽会增加学到的库的数量，但是会对性能造成影响
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub struct FindPackConfig {
  // 是否要根据相似度剪枝
  pub prune_similar: bool,
  // 是否学习trivial的表达式
  pub learn_trivial: bool,
}

impl Default for FindPackConfig {
  fn default() -> Self {
    Self {
      prune_similar: true,
      learn_trivial: true,
    }
  }
}

/// A Pareto Runner that uses Pareto optimization with beam search
pub struct ParetoRunner<Op, T, LA, LD>
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
    + OperationInfo,
  LA: Debug + Default + Clone,
  LD: Debug + Default + Clone,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + 'static
    + Display
    + FromStr,
  TypeSet<T>: ClassMatch,
  AstNode<Op>: TypeInfo<T>,
{
  /// The domain-specific rewrites to apply
  dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  /// dsr_rewrite_iters
  dsr_rewrite_iters: usize,
  /// BB query
  bb_query: BBQuery,
  // /// lib rewrites
  // lib_rewrites_with_condition:
  //   HashMap<usize, Vec<(Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>,
  // TypeMatch<T>)>>, /// lib exprs
  // past_exprs: HashMap<usize, RecExpr<AstNode<Op>>>,
  // past_libs: HashMap<usize, Pattern<AstNode<Op>>>,
  /// past lib_messages
  past_lib_message: ExpandMessage<Op, T>,
  /// Configuration for the beam search
  config: ParetoConfig<LA, LD>,
  /// lift_dsrs
  lift_dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  /// lower dsrs
  lower_dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  /// transform_dsrs
  transform_dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
}

impl<Op, T, LA, LD> ParetoRunner<Op, T, LA, LD>
where
  Op: Arity
    + OperationInfo
    + Teachable
    + Printable
    + Debug
    + Default
    + Display
    + Hash
    + Clone
    + Ord
    + Sync
    + Send
    + DiscriminantEq
    + 'static
    + BBInfo,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + 'static
    + Display
    + FromStr
    + TypeAnalysis,
  TypeSet<T>: ClassMatch,
  LA: Debug + Default + Clone,
  LD: Debug + Default + Clone,
  AstNode<Op>: TypeInfo<T> + Schedulable<LA, LD>,
{
  /// Create a new BeamRunner with the given domain-specific rewrites and
  /// configuration
  pub fn new<I>(
    dsrs: I,
    dsr_rewrite_iters: usize,
    bb_query: BBQuery,
    // lib_rewrites_with_condition: HashMap<
    //   usize,
    //   (Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>, TypeMatch<T>),
    // >,
    // past_exprs: HashMap<usize, RecExpr<AstNode<Op>>>,
    // past_libs: HashMap<usize, Pattern<AstNode<Op>>>,
    past_lib_message: ExpandMessage<Op, T>,
    config: ParetoConfig<LA, LD>,
  ) -> Self
  where
    I: IntoIterator<Item = Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  {
    // 如果USE_RULES为false，将dsrs清空
    let dsrs = dsrs.into_iter().collect();
    Self {
      dsrs,
      dsr_rewrite_iters,
      bb_query,
      // lib_rewrites_with_condition,
      // past_exprs,
      // past_libs,
      past_lib_message,
      config,
      lift_dsrs: vec![],
      lower_dsrs: vec![],
      transform_dsrs: vec![],
    }
  }

  pub fn with_vectorize_dsrs(
    &mut self,
    lift_dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
    lower_dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
    transform_dsrs: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  ) {
    self.lift_dsrs = lift_dsrs;
    self.lower_dsrs = lower_dsrs;
    self.transform_dsrs = transform_dsrs;
  }
  /// Run the e-graph and library learning process
  pub fn run_egraph(
    &self,
    roots: &[Id],
    egraph: EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  ) -> ParetoResult<Op, T> {
    // 首先将roots变成root，想EGraph中加入list节点实现
    let mut egraph = egraph;
    assert!(!roots.is_empty(), "Roots cannot be empty");
    let mut root = if roots.len() == 1 {
      roots[0]
    } else {
      let mut bbs = HashSet::new();
      for root in roots.iter() {
        bbs.extend(egraph[*root].data.bb.clone());
      }
      let bbs = bbs.into_iter().collect::<Vec<_>>();
      let mut list_op = AstNode::new(Op::list(), roots.iter().copied());
      list_op.operation_mut().set_bbs_info(bbs);
      egraph.add(list_op)
    };
    perf_infer::perf_infer(&mut egraph, &[root]);
    // 保存这个egraph，在向量化的时候需要
    let egraph_without_dsrs = egraph.clone();
    let timeout = Duration::from_secs(60 * 100_000);
    // let mut egraph = egraph.clone();
    // let root = egraph.add(AstNode::new(Op::list(), roots.iter().copied()));

    println!(
      "     • Initial egraph size: {}, eclasses: {}",
      egraph.total_size(),
      egraph.classes().len()
    );
    println!("     • After applying {} DSRs... ", self.dsrs.len());
    let start_time = Instant::now();
    let runner = EggRunner::<_, _, ()>::new(ISAXAnalysis::new(
      0,
      0,
      0,
      self.bb_query.clone(),
    ))
    .with_egraph(egraph)
    .with_time_limit(timeout)
    .with_iter_limit(self.dsr_rewrite_iters)
    .run(&self.dsrs);

    let mut origin_aeg = runner.egraph;
    let liblearn_egraph_size =
      (origin_aeg.classes().len(), origin_aeg.total_size());

    let cloned_egraph = origin_aeg.clone();
    for class in origin_aeg.classes_mut() {
      for node in class.nodes.iter_mut() {
        // 如果node的ty是空的，就将eclass的类型信息传给它
        if node.operation().get_result_type().is_empty() {
          let tys = cloned_egraph[class.id].data.get_type();
          node.operation_mut().set_result_type(tys.clone());
        }
      }
    }

    perf_infer::perf_infer(&mut origin_aeg, &[root]);

    println!(
      "       - Egraph size: {}, eclasses: {}",
      origin_aeg.total_size(),
      origin_aeg.classes().len()
    );

    // 接下来使用lib_rewrites_with_condition在进行重写, Analysis不能是空
    let past_lib_rewrites = self
      .past_lib_message
      .rewrites_conditions
      .values()
      .flat_map(|v| v.iter().map(|(r, _)| r.clone()))
      .collect::<Vec<_>>();

    // let runner = EggRunner::<_, _, ()>::new(ISAXAnalysis::empty())
    //   .with_egraph(origin_aeg.clone())
    //   .with_time_limit(timeout)
    //   .with_iter_limit(1)
    //   .run(&past_lib_rewrites);

    // 目前不会使用重写应用原来的lib，而是选择将past_exprs加入到原来的EGraph中，
    // 这样可以忽略掉不必要的lambda和apply节点

    let mut aeg = origin_aeg.clone();
    // aeg.dot().to_png("target/initial_egraph.png").unwrap();
    let mut aeg_roots = vec![root];

    for partial_expr in self.past_lib_message.get_exprs() {
      // println!("Adding past expr: {}", expr);

      aeg_roots.push(aeg.add_expr(&partial_expr));
    }

    // 进行BB信息的推断
    perf_infer::perf_infer(&mut aeg, &aeg_roots);

    // 在进行learn之前，先提取出之前库学习最大的lib，直接利用past_libs的keys就行
    let max_lib_id = self
      .past_lib_message
      .appliers
      .keys()
      .max()
      .map_or(0, |&id| id + 1); // past_libs的keys是从0开始的，所以+1

    let mut vectorized_liblearn_messages = vec![];
    let mut vectorized_egraph_with_root = None;

    // aeg.dot().to_png("target/initial_egraph.png").unwrap();
    if self.config.vectorize_config.vectorize {
      let vectorize_time = Instant::now();
      let (vecegraph_running_dsrs_with_root, lib_messages) = vectorize(
        aeg.clone(),
        root,
        &self.lift_dsrs,
        &self.lower_dsrs,
        &self.transform_dsrs,
        self.config.clone(),
        self.bb_query.clone(),
        max_lib_id,
      );
      origin_aeg = vecegraph_running_dsrs_with_root.0.clone();
      vectorized_liblearn_messages = lib_messages;
      root = vecegraph_running_dsrs_with_root.1;
      vectorized_egraph_with_root = Some(vecegraph_running_dsrs_with_root);
      println!(
        "     • Vectorized egraph in {}ms",
        vectorize_time.elapsed().as_millis()
      );
    }

    println!(
      "       - Final egraph size: {}, eclasses: {}",
      aeg.total_size(),
      aeg.classes().len()
    );

    info!(
      "Finished in {}ms; final egraph size: {}",
      start_time.elapsed().as_millis(),
      aeg.total_size()
    );

    // println!("Running co-occurrence analysis... ");
    // let co_time = Instant::now();
    // let co_ext = COBuilder::new(&aeg, roots);
    // let co_occurs = co_ext.run();
    // println!("Finished in {}ms", co_time.elapsed().as_millis());

    println!("      🧠 Anti-Unification Phase...");
    let au_time = Instant::now();

    // 如果启用了expand选项，那么就不走这一条路，
    // 直接使用expand的操作获取到所以用到的rewrites
    // 用于提前获取信息用于指导CostSet
    let mut lib_search_results = HashMap::new();

    let expand_message = if self.config.op_pack_config.enable_meta_au {
      expand(
        aeg.clone(),
        root,
        self.config.clone(),
        self.bb_query.clone(),
        max_lib_id,
      )
    } else {
      let liblearn_messages = if self.config.vectorize_config.vectorize {
        vectorized_liblearn_messages
      } else {
        // for ecls in aeg.classes() {
        //   println!("eclass id: {}, nodes: {:?}", ecls.id, ecls.nodes);
        // }
        let mut learned_lib = LearnedLibraryBuilder::default()
          .learn_constants(self.config.learn_constants)
          .max_arity(self.config.max_arity)
          // .with_co_occurs(co_occurs)
          .with_last_lib_id(max_lib_id)
          .with_liblearn_config(self.config.liblearn_config.clone())
          .with_ci_encoding_config(self.config.ci_encoding_config.clone())
          .with_clock_period(self.config.clock_period)
          .with_area_estimator(self.config.area_estimator.clone())
          .with_delay_estimator(self.config.delay_estimator.clone())
          .with_bb_query(self.bb_query.clone())
          .build(&aeg, aeg_roots);

        println!(
          "       - Found {} patterns in {}ms",
          learned_lib.size(),
          au_time.elapsed().as_millis()
        );
        println!("        • Deduplicating patterns... ");
        let dedup_time = Instant::now();
        learned_lib.deduplicate(&aeg);
        // for msg in learned_lib.messages() {
        //   let searcher = Pattern::from(msg.searcher_pe);
        //   let results = searcher.search(&origin_aeg);
        //   println!(
        //     "lib_id: {}, applier: {}, results.len: {}",
        //     msg.lib_id,
        //     Pattern::from(msg.applier_pe),
        //     results.len()
        //   );
        // }

        println!(
          "         • Deduplicated to {} patterns in {}ms",
          learned_lib.size(),
          dedup_time.elapsed().as_millis()
        );
        info!(
          "Reduced to {} patterns in {}ms",
          learned_lib.size(),
          dedup_time.elapsed().as_millis()
        );

        println!("        • learned {} libs", learned_lib.size());

        learned_lib.messages()
      };

      let mut rewrites_conditions = HashMap::new();
      let mut searchers = HashMap::new();
      let mut appliers = HashMap::new();
      for msg in liblearn_messages {
        let lib_id = msg.lib_id;
        rewrites_conditions
          .insert(lib_id, vec![(msg.rewrite.clone(), msg.condition)]);
        searchers.insert(lib_id, vec![msg.searcher_pe.clone()]);
        let applier: Pattern<_> = msg.applier_pe.into();
        appliers.insert(lib_id, applier);
        lib_search_results.insert(lib_id, msg.search_len);
      }

      ExpandMessage {
        rewrites_conditions,
        searchers,
        appliers,
        lib_search_results: lib_search_results.clone(),
        meta_libs: HashSet::new(),
      }
    };
    lib_search_results.extend(expand_message.lib_search_results.clone());

    // // FIXME: 复用性至上！！！
    // // 需要筛掉一些lib，如果所有lib对应的searcher匹配到的片段只有一个，
    // // 那么就删除这个lib
    // for lib in expand_message.libs() {
    //   let searchers = expand_message.searchers.get(&lib).unwrap();
    //   let mut cnt = 0;
    //   for searcher in searchers {
    //     let searcher = Pattern::from(searcher.clone());
    //     let results = searcher.search(&aeg);
    //     cnt += results.len();
    //     if cnt > 1 {
    //       break;
    //     }
    //   }
    //   if cnt <= 1 {
    //     expand_message.delete_lib(lib);
    //   }
    // }

    println!("      🔁 Adding Libs + Beam Search...");
    // let mut vec_applier = Vec::new();
    // for (i, applier) in expand_message.appliers.iter() {
    //   vec_applier.push((i.clone(), applier.clone()));
    // }
    // vec_applier.sort_by(|a, b| a.0.cmp(&b.0));
    // for (lib_id, applier) in vec_applier {
    //   println!("lib_id: {}, applier: {}", lib_id, applier);
    // }

    // rewrites_conditions中项的个数表示lib的数目，
    // vec中元素总个数表明rewrite的数目
    let num_libs = expand_message.rewrites_conditions.len();
    let num_rewrites = expand_message
      .rewrites_conditions
      .values()
      .map(|v| v.len())
      .sum::<usize>();

    println!(
      "       • There are {} rewrites and {} libs",
      num_rewrites, num_libs
    );
    let lib_rewrite_time = Instant::now();

    let mut new_all_rewrites = expand_message
      .rewrites_conditions
      .values()
      .flat_map(|v| v.iter().map(|(r, _)| r.clone()))
      .collect::<Vec<_>>();
    // 将past_lib_rewrites加入，同台竞技，使用的是origin_aeg
    new_all_rewrites.extend(past_lib_rewrites);
    // deduplicate the rewrites
    new_all_rewrites.sort_by(|a, b| a.name.cmp(&b.name));
    new_all_rewrites.dedup_by(|a, b| a.name == b.name);

    println!(
      "       • after extend, there are {} rewrites",
      new_all_rewrites.len()
    );
    println!(
      "         • before running lib rewrites, egraph size: {}, eclasses: {}",
      origin_aeg.total_size(),
      origin_aeg.classes().len()
    );
    // println!("rewrites: {:#?}", new_all_rewrites);
    // for eclass in origin_aeg.classes() {
    //   println!("eclass: {:?}", eclass);
    // }
    // origin_aeg的analysi带上lib_search_results
    origin_aeg
      .analysis
      .with_search_len_map(lib_search_results.clone());
    let runner = EggRunner::<_, _, ()>::new(ISAXAnalysis::new(
      self.config.final_beams,
      self.config.inter_beams,
      self.config.lps,
      self.bb_query.clone(),
    ))
    .with_egraph(origin_aeg.clone())
    .with_iter_limit(1)
    .with_time_limit(timeout)
    .with_node_limit(1_000_000)
    .run(&new_all_rewrites);

    let mut egraph = runner.egraph;
    // for rewrite in new_all_rewrites {
    //   println!("Running rewrite: {}", rewrite.name);
    //   let matches = rewrite.search(&origin_aeg);
    //   println!("There are {} matches for {:?}", matches.len(), rewrite);
    //   println!("matches: {:#?}", matches);
    // }
    perf_infer::perf_infer(&mut egraph, &[root]);
    println!("         • Rebuilding egraph after lib rewrites...");
    Self::recalculate_all_data(&mut egraph, root);
    egraph.rebuild();

    println!(
      "         • Final egraph size: {}, eclasses: {}",
      egraph.total_size(),
      egraph.classes().len()
    );

    // egraph.dot().to_png("target/final_egraph.png").unwrap();
    // println!("roots: {:?}", roots);
    // for ecls in egraph.classes() {
    //   println!(
    //     "eclass id: {}, size: {}, nodes: {:?}",
    //     ecls.id,
    //     ecls.nodes.len(),
    //     ecls.nodes
    //   );
    //   for node in ecls.nodes.clone() {
    //     if node.operation().is_lib() {
    //       println!("lib node: {:?}", node);
    //     }
    //   }
    // }

    let isax_cost = egraph[egraph.find(root)].data.clone();
    // for ecls in egraph.classes() {
    //   println!(
    //     "eclass {}: nodes: {:?}, cs: {:?}",
    //     ecls.id, ecls.nodes, ecls.data.cs
    //   );
    // }

    // println!("root: {}", root);
    // let args1 = egraph[egraph.find(root)].nodes[0].args();
    // let args2 = egraph[egraph.find(root)].nodes[1].args();
    // for arg in args1 {
    //   println!("arg1: {:#?}", egraph[*arg].data.cs);
    // }
    // println!("the second");
    // for arg in args2 {
    //   println!("arg2: {:#?}", egraph[*arg].data.cs);
    // }

    // 打印root的id和cs

    // println!("root_vec: {:?}", root_vec);

    info!("Finished in {}ms", lib_rewrite_time.elapsed().as_millis());
    info!("Stop reason: {:?}", runner.stop_reason.unwrap());
    info!("Number of nodes: {}", egraph.total_size());

    // egraph.dot().to_png("target/foo.png").unwrap();

    // for ecls in egraph.classes() {
    //   // if ecls.nodes.iter().any(|n| n.operation().is_lib()) {
    //   println!(
    //     "ecls: {}, nodes: {:?}, cs: {:?}",
    //     ecls.id, ecls.nodes, ecls.data.cs
    //   );
    //   // println!("cs: {:?}", ecls.data.cs);
    //   // }
    // }
    // panic!("Debugging egraph");
    // println!("learned libs");
    // let all_libs: Vec<_> = learned_lib.libs().collect();
    // println!("cs: {:#?}", isax_cost.cs);
    let mut extract_results = Vec::new();
    let mut chosen_rewrites = Vec::new();
    let mut chosen_libids = HashSet::new();
    // 存储每个lib选择的重写和库
    let mut lib_message = ExpandMessage::default();
    for i in 0..isax_cost.cs.set.len() {
      let mut chosen_rewrites_per_libsel = vec![];
      let mut chosen_libs_per_libsel: HashMap<usize, Pattern<AstNode<Op>>> =
        HashMap::new();
      let mut chosen_searchers_per_libsel: HashMap<
        usize,
        Pattern<AstNode<Op>>,
      > = HashMap::new();
      for lib in &isax_cost.cs.set[i].libs {
        // println!("lib: {}, max_lib_id: {}", lib.0.0, max_lib_id);
        if lib.0 .0 < max_lib_id {
          // 从self.lib_rewrites中取出
          // 打印self.lib_rewrites
          // println!("{}: {:?}", lib.0.0, self.lib_rewrites_with_condition);

          chosen_rewrites_per_libsel.extend(
            self
              .past_lib_message
              .rewrites_conditions
              .get(&lib.0 .0)
              .unwrap()
              .iter()
              .map(|(r, _)| r.clone())
              .clone(),
          );
          chosen_libs_per_libsel.insert(
            lib.0 .0,
            self.past_lib_message.appliers[&lib.0 .0].clone(),
          );
          if let Some(searcher) = self
            .past_lib_message
            .searchers
            .get(&lib.0 .0)
            .and_then(|searchers| searchers.first())
          {
            chosen_searchers_per_libsel.insert(lib.0 .0, searcher.clone().into());
          }
          lib_message.insert_from_messages(lib.0 .0, &self.past_lib_message);
        } else {
          // let new_lib = lib.0.0 - max_lib_id;
          // chosen_rewrites.push(lib_rewrites[new_lib].clone());
          // learned_libs.push((lib.0.0, libs[new_lib].clone()));
          // rewrites_map.insert(
          //   lib.0.0,
          //   (
          //     lib_rewrites[new_lib].clone(),
          //     rewrite_conditions[new_lib].clone(),
          //   ),
          // );

          // println!("new_lib: {}", new_lib);
          chosen_rewrites_per_libsel.extend(
            expand_message
              .rewrites_conditions
              .get(&lib.0 .0)
              .unwrap()
              .iter()
              .map(|(r, _)| r.clone())
              .clone(),
          );
          chosen_libs_per_libsel
            .insert(lib.0 .0, expand_message.appliers[&lib.0 .0].clone());
          if let Some(searcher) = expand_message
            .searchers
            .get(&lib.0 .0)
            .and_then(|searchers| searchers.first())
          {
            chosen_searchers_per_libsel.insert(lib.0 .0, searcher.clone().into());
          }
          // println!(
          //   "choose: {}",
          //   Pattern::from(expand_message.libs[&lib.0.0].0.clone())
          // );
          lib_message.insert_from_messages(lib.0 .0, &expand_message);

          // } else {
          //   // 说明是一个meta lib
          //   println!("        • We have leaned a meta lib !");
          //   chosen_rewrites_per_libsel
          //     .extend(expand_message.meta_au_rewrites[&lib.0.0].clone());
          //   learned_libs.push((lib.0.0,
          // expand_message.libs[&lib.0.0].clone()));
          //   chosen_libs_per_libsel
          //     .insert(lib.0.0, expand_message.libs[&lib.0.0].clone().1);
          //   for rewrite in expand_message.meta_au_rewrites[&lib.0.0].clone()
          // {     rewrites_map.insert(
          //       lib.0.0,
          //       (rewrite.clone(),
          // expand_message.conditions[&lib.0.0].clone()),     );
          //   }
          // }
        }
      }
      chosen_rewrites.extend(chosen_rewrites_per_libsel.clone());

      // 更新annotated_egraphs
      // annotated_egraphs.push(AnnotatedEGraph::new(
      //   origin_aeg.clone(),
      //   chosen_rewrites_per_libsel,
      //   chosen_libs_per_libsel,
      //   root,
      //   isax_cost.cs.set[i].cycles.into_inner() as usize,
      //   isax_cost.cs.set[i].area,
      // ));

      let runner = EggRunner::<_, _, ()>::new(ISAXAnalysis::new(
        self.config.final_beams,
        self.config.inter_beams,
        self.config.lps,
        self.bb_query.clone(),
      ))
      .with_egraph(origin_aeg.clone())
      .with_iter_limit(1)
      .with_time_limit(timeout)
      .with_node_limit(1_000_000)
      .run(&chosen_rewrites_per_libsel);
      let mut aeg = runner.egraph;
      perf_infer::perf_infer(&mut aeg, &vec![root]);

      // 在此之前，为aeg带上准确的lat_acc，用于ILP
      let mut exact_lat_acc_map = HashMap::new();
      let scheduler = Scheduler::new(
        self.config.clock_period,
        self.config.area_estimator.clone(),
        self.config.delay_estimator.clone(),
        self.bb_query.clone(),
      );
      for ecls in aeg.classes() {
        for node in ecls.nodes.iter() {
          if node.operation().is_lib() {
            let mut bbs = node.operation().get_bbs_info();
            bbs.sort();
            let lib_id = node.operation().get_libid();
            if !exact_lat_acc_map.contains_key(&(lib_id, bbs.clone())) {
              // Recompute exact latency from the library body/searcher instead
              // of the applier. Appliers contain Lambda/Apply/Lib wrappers,
              // which are rewrite templates rather than schedulable datapaths.
              let Some(searcher_expr) = chosen_searchers_per_libsel.get(&lib_id)
              else {
                continue;
              };
              let new_expr = searcher_expr
                .clone()
                .ast
                .iter()
                .map(|node| match node {
                  egg::ENodeOrVar::ENode(ast_node) => {
                    let mut new_ast_node = ast_node.clone();
                    // 更新正确的bbs信息
                    new_ast_node.operation_mut().set_bbs_info(bbs.clone());
                    new_ast_node
                  }
                  egg::ENodeOrVar::Var(_) => Op::var(0),
                })
                .collect::<Vec<AstNode<Op>>>();
              let new_expr = RecExpr::from(new_expr);
              let (exact_lat_cpu, lat_acc, exact_area) =
                scheduler.asap_schedule(&new_expr);
              let default = node
                .as_binding_expr()
                .and_then(|binding| match binding {
                  BindingExpr::Lib(_, _, _, lat_cpu, default_lat_acc, area) => {
                    Some((lat_cpu.0, default_lat_acc.0, area))
                  }
                  _ => None,
                });
              if std::env::var_os("ISAEGG_DUMP_EXACT_LAT").is_some()
                && lat_acc == 0.0
              {
                println!(
                  "ISAEGG_EXACT_LAT_ZERO lib={} bbs={:?} default={:?} exact_lat_cpu={} exact_lat_acc={} exact_area={} expr={:?}",
                  lib_id, bbs, default, exact_lat_cpu, lat_acc, exact_area, new_expr
                );
              }
              let lat_acc = if lat_acc > 0.0 && exact_area > 0 {
                lat_acc
              } else {
                default.map(|(_, default_lat_acc, _)| default_lat_acc).unwrap_or(lat_acc)
              };
              exact_lat_acc_map.insert((lib_id, bbs), lat_acc);
            }
          }
        }
      }
      // 打印extract_lat_acc_map
      // println!("extract_lat_acc_map: {:#?}", exact_lat_acc_map);
      aeg.analysis.with_lat_map(exact_lat_acc_map.clone());

      // let mut extractor = LibExtractor::new(&aeg, self.bb_query.clone());
      // let best = extractor.best(annotated_egraph.root);

      // for ecls in aeg.classes() {
      //   println!("eclass {}: nodes: {:?}", ecls.id, ecls.nodes,);
      // }

      let lp_cf = ISAXLpCF::new(self.bb_query.clone());
      aeg = eliminate_lambda(&aeg);
      let mut lp_extractor = LpExtractor::new(&aeg, lp_cf);
      let best = lp_extractor.solve(root);
      // println!("best solution:");
      // for (id, node) in best.iter().enumerate() {
      //   println!("  {}: {:?}", id, node);
      // }
      // 取出来最终表达式之后，取出真正选择的lib_id
      let mut lib_reuse_map = HashMap::new();
      for node in best.iter() {
        if node.operation().is_lib() {
          let lib_id = node.operation().get_libid();
          // 记录重用次数
          *lib_reuse_map.entry(lib_id).or_insert(0) += 1;
        }
      }
      chosen_libs_per_libsel
        .retain(|lib_id, _| lib_reuse_map.get(lib_id).is_some());
      let (cycles, area) =
        rec_cost(&best, &self.bb_query, exact_lat_acc_map.clone());
      if std::env::var_os("ISAEGG_DUMP_EXTRACT").is_some() {
        println!("--- ISAEGG_DUMP_EXTRACT solution {i} expr ---");
        for (node_id, node) in best.iter().enumerate() {
          println!("dump_expr node={node_id} {:?}", node);
        }
        println!("--- ISAEGG_DUMP_EXTRACT solution {i} selected_libs ---");
        let mut selected_lib_ids = chosen_libs_per_libsel
          .keys()
          .copied()
          .collect::<Vec<_>>();
        selected_lib_ids.sort();
        for lib_id in selected_lib_ids {
          if let Some(searcher) = chosen_searchers_per_libsel.get(&lib_id) {
            println!("dump_lib lib={lib_id} searcher={searcher}");
          }
          if let Some(applier) = chosen_libs_per_libsel.get(&lib_id) {
            println!("dump_lib lib={lib_id} applier={applier}");
          }
        }
        println!("--- ISAEGG_DUMP_EXTRACT solution {i} rec_cost ---");
        println!(
          "{}",
          dump_rec_cost::<Op, LA, LD>(
            &best,
            &self.bb_query,
            exact_lat_acc_map.clone()
          )
        );
        println!("--- ISAEGG_DUMP_EXTRACT solution {i} end ---");
      }
      let func_cycles =
        cycles_for_every_function(&best, &self.bb_query, exact_lat_acc_map);
      // 组装成一个ExtractResult
      let es = ExtractResult::new(
        best,
        chosen_libs_per_libsel,
        cycles,
        area,
        func_cycles,
        lib_reuse_map,
      );
      chosen_libids.extend(es.libs.keys().cloned());
      extract_results.push(es);
    }

    // 使用chosen_libids过滤lib_message
    lib_message.retain_with_ids(&chosen_libids);
    let chosen_meta_libids = chosen_libids
      .iter()
      .filter(|lib_id| {
        self.past_lib_message.meta_libs.contains(lib_id)
          || expand_message.meta_libs.contains(lib_id)
      })
      .cloned()
      .collect::<HashSet<_>>();
    println!("         • chosen_meta_libs: {:?}", chosen_meta_libids);

    // deduplicate chosen_rewrites
    chosen_rewrites.sort_unstable_by_key(|r| r.name.clone());
    chosen_rewrites.dedup_by_key(|r| r.name.clone());

    println!("         • chosen_rewrites: {}", chosen_rewrites.len());

    debug!("upper bound ('full') cost: {}", isax_cost.cs.set[0].cycles);

    // println!("annotated_egraphs: {}", annotated_egraphs.len());
    // message.insert(
    //   "extract_time".to_string(),
    //   format!("{}ms", ex_time.elapsed().as_millis()).to_string(),
    // );
    println!("         • Extracted {} results.", extract_results.len());

    for (i, es) in extract_results.iter().enumerate() {
      println!(
        "           • es{}: cycles: {}, area: {}",
        i, es.cycles, es.area
      );
    }
    ParetoResult {
      num_libs: chosen_rewrites.len(),
      lib_message,
      extract_results,
      vectorized_egraph_with_root: vectorized_egraph_with_root,
      run_time: start_time.elapsed(),
      egraph_size: liblearn_egraph_size,
    }
  }

  fn recalculate_all_data(
    egraph: &mut EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
    id: Id,
  ) {
    let mut visited = HashSet::new();
    Self::recalculate_all_data_with_visited(egraph, id, &mut visited);
  }

  fn recalculate_all_data_with_visited(
    egraph: &mut EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
    id: Id,
    visited: &mut HashSet<Id>,
  ) {
    if visited.contains(&id) {
      return;
    }
    visited.insert(id);

    let eclass = &egraph[id];
    let nodes = eclass.nodes.clone();
    // Recalculate the data for this eclass
    let mut new_data = ISAXCost::empty();
    for (i, node) in nodes.iter().enumerate() {
      for arg in node.args() {
        Self::recalculate_all_data_with_visited(egraph, *arg, visited);
      }
      let data = ISAXAnalysis::make(egraph, node);
      if i == 0 {
        // The first node is the root node, we use its data as the base
        new_data = data;
      } else {
        // Merge the data of the other nodes into the first one
        // This is necessary to ensure that all nodes contribute to the eclass
        // data and that we can use it for further analysis
        egraph.analysis.merge(&mut new_data, data);
      }
    }
    let eclass = &mut egraph[id];
    eclass.data = new_data;
  }
}

impl<Op, T, LA, LD> BabbleParetoRunner<Op, T> for ParetoRunner<Op, T, LA, LD>
where
  Op: Arity
    + OperationInfo
    + Teachable
    + Printable
    + Debug
    + Default
    + Display
    + Hash
    + Clone
    + Ord
    + Sync
    + Send
    + DiscriminantEq
    + 'static
    + BBInfo,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + 'static
    + Display
    + FromStr
    + TypeAnalysis,
  TypeSet<T>: ClassMatch,
  LA: Debug + Default + Clone,
  LD: Debug + Default + Clone,
  AstNode<Op>: TypeInfo<T> + Schedulable<LA, LD>,
{
  fn run(&self, expr: Expr<Op>) -> ParetoResult<Op, T> {
    self.run_multi(vec![expr])
  }

  fn run_multi(&self, exprs: Vec<Expr<Op>>) -> ParetoResult<Op, T> {
    // First, let's turn our list of exprs into a list of recexprs
    let recexprs: Vec<RecExpr<AstNode<Op>>> =
      exprs.into_iter().map(RecExpr::from).collect();
    let mut egraph = EGraph::new(ISAXAnalysis::new(
      self.config.final_beams,
      self.config.inter_beams,
      self.config.lps,
      self.bb_query.clone(),
    ));
    let roots = recexprs
      .iter()
      .map(|x| egraph.add_expr(x))
      .collect::<Vec<_>>();
    egraph.rebuild();

    if self.config.enable_widths_merge {
      // 加入类型重写，相当于是给egraph中添加node并做合并
      let widths = vec![8 as u32, 16 as u32, 32 as u32, 64 as u32];
      let mut int_width: HashMap<i64, HashSet<u32>> = HashMap::new();
      let mut ecls_ids: HashMap<i64, HashSet<Id>> = HashMap::new();
      // 收集已经存在的位宽信息和eclass信息
      for ecls in egraph.classes() {
        for node in ecls.nodes.clone() {
          if let Some((a, aw)) = node.operation().get_const() {
            if !int_width.contains_key(&a) {
              int_width.insert(a, HashSet::new());
              ecls_ids.insert(a, HashSet::new());
            }
            let int_node = int_width.get_mut(&a).unwrap();
            int_node.insert(aw);
            let ecls_id = ecls_ids.get_mut(&a).unwrap();
            ecls_id.insert(ecls.id);
          }
        }
      }
      // 遍历width，如果已经存在的位宽不包含当前的位宽，则添加
      for width in widths.iter() {
        for (a, aw) in int_width.clone().iter() {
          if !aw.contains(width) {
            let new_node = AstNode::leaf(Op::make_const((*a, *width)));
            let new_ecls_id = egraph.add_uncanonical(new_node);
            // 将当前的eclass添加到ecls_ids中
            let ecls_id = ecls_ids.get_mut(a).unwrap();
            ecls_id.insert(new_ecls_id);
          }
        }
      }
      // 遍历ecls_ids，进行union
      for (_, ecls_id) in ecls_ids.iter() {
        let mut ids = Vec::new();
        for id in ecls_id.iter() {
          ids.push(*id);
        }
        if ids.len() > 1 {
          let first = ids[0];
          for i in 1..ids.len() {
            egraph.union(first, ids[i]);
          }
        }
      }
      egraph.rebuild();
    }

    self.run_egraph(&roots, egraph)
  }

  fn run_equiv_groups(
    &self,
    expr_groups: Vec<Vec<Expr<Op>>>,
  ) -> ParetoResult<Op, T> {
    // First, let's turn our list of exprs into a list of recexprs
    let recexpr_groups: Vec<Vec<_>> = expr_groups
      .into_iter()
      .map(|group| group.into_iter().map(RecExpr::from).collect())
      .collect();

    let mut egraph = EGraph::new(ISAXAnalysis::new(
      self.config.final_beams,
      self.config.inter_beams,
      self.config.lps,
      self.bb_query.clone(),
    ));

    let roots: Vec<_> = recexpr_groups
      .into_iter()
      .map(|mut group| {
        let first_expr = group.pop().unwrap();
        let root = egraph.add_expr(&first_expr);
        for expr in group {
          let class = egraph.add_expr(&expr);
          egraph.union(root, class);
        }
        root
      })
      .collect();

    egraph.rebuild();
    self.run_egraph(&roots, egraph)
  }
}

// Implement Debug that doesn't depend on Op being Debug
impl<Op, T, LA, LD> std::fmt::Debug for ParetoRunner<Op, T, LA, LD>
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
    + OperationInfo,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + 'static
    + Display
    + FromStr
    + TypeAnalysis,
  LA: Debug + Default + Clone,
  LD: Debug + Default + Clone,
  TypeSet<T>: ClassMatch,
  AstNode<Op>: TypeInfo<T>,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("BeamRunner")
      .field("dsrs_count", &self.dsrs.len())
      .field("config", &self.config)
      .finish()
  }
}
