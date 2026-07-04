//! The primary interface for library learning through antiunification.
//!
//! The antiunification algorithm is as follows: For each pair of eclasses (a,
//! b), we store a set of partial expressions AU(a, b). Partial expressions are
//! expressions where some of the sub-expressions may be replaced by another
//! data type. For example, a pattern is (equivalent to) a partial expression
//! over variables. In this algorithm, we use partial expressions over pairs of
//! enodes.
//!
//! To compute the set AU(a, b): For each pair of enodes with matching operators
//! and arities op(x1, ..., xn) \in a and op(y1, ..., yn) \in b, we recursively
//! compute AU(x1, y1), ..., AU(xn, yn). Then, for every n-tuple of partial
//! expressions (z1, ..., zn) with z1 \in AU(x1, y1), ..., zn \in AU(xn, yn), we
//! add the partial expression op(z1, ..., zn) to the set AU(a, b).
//! If the set AU(a, b) is empty, we add to it the partial expression (a, b).
use crate::au_filter::{io_filter, max_inputs, CiEncodingConfig, TypeAnalysis};
use crate::bb_query::{self, BBQuery};
use crate::expand::MetaAUConfig;
use crate::extract::beam_pareto::{
  compute_full_hash, compute_hash_level, ISAXAnalysis, LevelConflictState,
  StructuralHash, TypeInfo, TypeSet,
};
use crate::rewrites::TypeMatch;
// 使用随机数
use crate::runner::{
  AUMergeMod, EnumMode, FindPackConfig, LiblearnConfig, LiblearnCost,
  OperationInfo,
};
use crate::{
  analysis::SimpleAnalysis,
  ast_node::{Arity, AstNode, Expr, PartialExpr},
  co_occurrence::CoOccurrences,
  dfta::Dfta,
  extract::beam_pareto::ClassMatch,
  schedule::{Schedulable, Scheduler},
  teachable::{BindingExpr, Teachable},
  COBuilder,
};
use crate::{ast_node, vectorize};
use bitvec::prelude::*;
use egg::{
  Analysis, ConditionalApplier, EGraph, Id, Language, LanguageMapper, Pattern,
  RecExpr, Rewrite, Runner, Searcher, SimpleLanguageMapper, Subst, Var,
};
use itertools::Itertools;
use lexpr::print;
use log::{debug, info};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::ops;
use std::sync::mpsc;
use std::time::Duration;
use std::{
  collections::{BTreeMap, BTreeSet, HashMap},
  fmt::{Debug, Display},
  num::ParseIntError,
  str::FromStr,
};
use std::{hash::Hash, hash::Hasher, time::Instant, vec};
use thiserror::Error;

use crate::au_merge::AUMerger;

use rayon::prelude::*;

#[derive(Debug, Default)]
struct MetaAUMaskProfile {
  opmask_count: usize,
  size: usize,
  has_nested_opmask: bool,
  root_is_opmask: bool,
  has_stable_context_op: bool,
}

fn meta_au_mask_profile<Op, T>(expr: &PartialExpr<Op, T>) -> MetaAUMaskProfile
where
  Op: OperationInfo + Clone + Ord,
  T: Clone + Ord,
{
  fn visit<Op, T>(
    expr: &PartialExpr<Op, T>,
    inside_opmask: bool,
    is_root: bool,
    profile: &mut MetaAUMaskProfile,
  ) where
    Op: OperationInfo + Clone + Ord,
    T: Clone + Ord,
  {
    profile.size += 1;
    match expr {
      PartialExpr::Node(node) => {
        let is_opmask = node.operation().is_opmask();
        if is_opmask {
          profile.opmask_count += 1;
          profile.has_nested_opmask |= inside_opmask;
          profile.root_is_opmask |= is_root;
        } else {
          profile.has_stable_context_op = true;
        }
        for child in node.iter() {
          visit(child, inside_opmask || is_opmask, false, profile);
        }
      }
      PartialExpr::Hole(_) => {}
    }
  }

  let mut profile = MetaAUMaskProfile::default();
  visit(expr, false, true, &mut profile);
  profile
}

fn meta_au_mask_filter_reason(
  profile: &MetaAUMaskProfile,
  config: MetaAUConfig,
) -> Option<&'static str> {
  if profile.opmask_count > config.max_opmask_per_meta_au {
    return Some("too_many_opmask");
  }
  if profile.opmask_count <= 1 {
    return None;
  }
  if !config.allow_nested_opmask && profile.has_nested_opmask {
    return Some("nested_opmask");
  }
  if profile.root_is_opmask {
    return Some("root_opmask");
  }
  if !profile.has_stable_context_op {
    return Some("no_stable_context");
  }
  if profile.size > 0
    && (profile.opmask_count as f64 / profile.size as f64)
      > config.max_opmask_ratio
  {
    return Some("opmask_ratio");
  }
  None
}

/// A library function's name.
#[derive(
  Debug,
  Clone,
  Copy,
  PartialEq,
  Eq,
  PartialOrd,
  Ord,
  Hash,
  Serialize,
  Deserialize,
)]
pub struct LibId(pub usize);

impl Display for LibId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "l{}", self.0)
  }
}

/// An error when parsing a [`LibId`].
#[derive(Clone, Debug, Error)]
pub enum ParseLibIdError {
  /// The string did not start with "$"
  #[error("expected de Bruijn index to start with 'l")]
  NoLeadingL,
  /// The index is not a valid unsigned integer
  #[error(transparent)]
  InvalidIndex(ParseIntError),
}

impl FromStr for LibId {
  type Err = ParseLibIdError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    if let Some(n) = s.strip_prefix('l') {
      let n = n.parse().map_err(ParseLibIdError::InvalidIndex)?;
      Ok(LibId(n))
    } else {
      Err(ParseLibIdError::NoLeadingL)
    }
  }
}

/// Signature of a pattern match in an e-graph.
/// Used to deduplicate equivalent patterns.
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Hash)]
pub struct Match {
  /// The e-class that the match is found in.
  class: Id,
  /// The range of the match's substitution
  /// (a multiset of class ids, stores as a sorted vector).
  actuals: Vec<Id>,
}

impl Match {
  pub fn new(class: Id, mut actuals: Vec<Id>) -> Self {
    actuals.sort();
    Self { class, actuals }
  }
}
// 用来保存AU和其对应的Match
#[derive(Debug, Clone)]
pub struct AU<Op: OperationInfo + Clone + Ord, T: Clone + Ord, Type> {
  /// The anti-unification
  expr: PartialExpr<Op, T>,
  /// delay
  delay: usize,
  /// latency gain
  latency_gain: f64,
  /// area_cost
  area_cost: usize,
  /// strategy for sort
  liblearn_cost: LiblearnCost,
  /// search len
  search_len: usize,
  /// Type info
  _phantom: std::marker::PhantomData<Type>,
}

impl<
    Op: PartialEq + OperationInfo + Clone + Ord,
    T: PartialEq + Clone + Ord,
    Type,
  > PartialEq for AU<Op, T, Type>
{
  fn eq(&self, other: &Self) -> bool {
    self.expr == other.expr
  }
}

impl<Op: Eq + OperationInfo + Clone + Ord, T: Eq + Clone + Ord, Type> Eq
  for AU<Op, T, Type>
{
}

impl<Op: Hash + OperationInfo + Clone + Ord, T: Hash + Clone + Ord, Type> Hash
  for AU<Op, T, Type>
{
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.expr.hash(state);
  }
}

impl<Op: OperationInfo + Clone + Ord, T: Clone + Ord, Type> AU<Op, T, Type>
where
  Type: Debug
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
  TypeSet<Type>: ClassMatch,
  AstNode<Op>: TypeInfo<Type>,
{
  pub fn new(
    expr: PartialExpr<Op, T>,
    delay: usize,
    latency_gain: f64,
    area_cost: usize,
    liblearn_cost: LiblearnCost,
  ) -> Self {
    Self {
      expr,
      delay,
      latency_gain,
      area_cost,
      liblearn_cost,
      search_len: 0,
      _phantom: std::marker::PhantomData,
    }
  }
  pub fn new_with_expr(
    expr: PartialExpr<Op, T>,
    liblearn_cost: LiblearnCost,
  ) -> Self
  where
    Op: Clone
      + Arity
      + Debug
      + Default
      + Display
      + Ord
      + Send
      + Sync
      + Teachable
      + 'static
      + Hash,
    AstNode<Op>: Language,
    T: Clone + Debug + Hash + Ord + Default,
  {
    let delay = expr.get_delay();
    Self {
      expr,
      delay,
      latency_gain: 0.0,
      area_cost: 0,
      liblearn_cost,
      search_len: 0,
      _phantom: std::marker::PhantomData,
    }
  }
}

impl<Op: OperationInfo + Clone + Ord, T: Clone + Ord, Type> AU<Op, T, Type> {
  pub fn expr(&self) -> &PartialExpr<Op, T> {
    &self.expr
  }

  pub fn delay(&self) -> usize {
    self.delay
  }

  pub fn latency_gain(&self) -> f64 {
    self.latency_gain
  }

  pub fn area_cost(&self) -> usize {
    self.area_cost
  }

  pub fn liblearn_cost(&self) -> LiblearnCost {
    self.liblearn_cost
  }

  pub fn set_search_len(&mut self, search_len: usize) {
    self.search_len = search_len;
  }
}
// 为AU实现Ord，只对比matches的大小
impl<Op: Eq + OperationInfo + Clone + Ord, T: Eq + Clone + Ord, Type> PartialOrd
  for AU<Op, T, Type>
{
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    // Some(self.matches.cmp(&other.matches))
    // Some(self.expr.size().cmp(&other.expr.size()))
    // Some(self.delay.cmp(&other.delay))
    let self_holes = self.expr.num_holes();
    let other_holes = other.expr.num_holes();
    let ord = match self.liblearn_cost {
      LiblearnCost::Size => self.expr.size().cmp(&other.expr.size()),
      LiblearnCost::Delay => self.delay.cmp(&other.delay),
      LiblearnCost::LatencyGainArea => {
        self.area_cost.cmp(&other.area_cost).then(
          OrderedFloat(other.latency_gain)
            .cmp(&OrderedFloat(self.latency_gain)),
        )
      } // area从小到大，延迟增益从大到小
    }
    .then(other_holes.cmp(&self_holes))
    .then(self.expr.cmp(&other.expr));
    Some(ord)
  }
}

impl<Op: Eq + OperationInfo + Clone + Ord, T: Eq + Clone + Ord, Type> Ord
  for AU<Op, T, Type>
{
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    // self.matches.cmp(&other.matches)
    // self.expr.size().cmp(&other.expr.size())
    // self.delay.cmp(&other.delay)
    let self_holes = self.expr.num_holes();
    let other_holes = other.expr.num_holes();
    match self.liblearn_cost {
      LiblearnCost::Size => self.expr.size().cmp(&other.expr.size()),
      LiblearnCost::Delay => self.delay.cmp(&other.delay),
      LiblearnCost::LatencyGainArea => {
        self.area_cost.cmp(&other.area_cost).then(
          OrderedFloat(other.latency_gain)
            .cmp(&OrderedFloat(self.latency_gain)),
        )
      }
    }
    .then(other_holes.cmp(&self_holes))
    .then(self.expr.cmp(&other.expr))
  }
}

#[derive(Clone, Debug)]
pub struct LearnedLibraryBuilder<Op, Type, LA, LD>
where
  Op: Arity
    + Clone
    + Debug
    + Ord
    + Sync
    + Send
    + Display
    + Hash
    + DiscriminantEq
    + 'static
    + Teachable
    + OperationInfo,
  Type: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + 'static
    + Display,
  LA: Debug + Clone + Default,
  LD: Debug + Clone + Default,
  AstNode<Op>: TypeInfo<Type>,
{
  _phantom: std::marker::PhantomData<Type>,
  learn_trivial: bool,
  learn_constants: bool,
  max_arity: Option<usize>,
  banned_ops: Vec<Op>,
  roots: Vec<Id>,
  co_occurences: Option<CoOccurrences>,
  last_lib_id: usize,
  clock_period: usize,
  area_estimator: LA,
  delay_estimator: LD,
  bb_query: BBQuery,
  liblearn_config: LiblearnConfig,
  meta_au_config: MetaAUConfig,
  ci_encoding_config: CiEncodingConfig,
  /// 是否在包搜索模式，在这个模式下，需要放松一些限制
  find_packs: bool,
  /// 包搜索模式的配置
  find_pack_config: FindPackConfig,
}

impl<Op, Type, LA, LD> Default for LearnedLibraryBuilder<Op, Type, LA, LD>
where
  Op: Arity
    + Clone
    + Debug
    + Ord
    + Sync
    + Send
    + Display
    + Hash
    + DiscriminantEq
    + 'static
    + Teachable
    + OperationInfo,
  Type: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + 'static
    + Display,
  LA: Debug + Clone + Default,
  LD: Debug + Clone + Default,
  AstNode<Op>: TypeInfo<Type>,
{
  fn default() -> Self {
    Self {
      _phantom: std::marker::PhantomData,
      learn_trivial: false,
      learn_constants: false,
      max_arity: None,
      banned_ops: vec![],
      roots: vec![],
      co_occurences: None,
      last_lib_id: 0,
      clock_period: 1000,
      area_estimator: LA::default(),
      delay_estimator: LD::default(),
      bb_query: BBQuery::default(),
      liblearn_config: LiblearnConfig::default(),
      meta_au_config: MetaAUConfig::default(),
      ci_encoding_config: CiEncodingConfig::default(),
      find_packs: false,
      find_pack_config: FindPackConfig::default(),
    }
  }
}

impl<Op, Type, LA, LD> LearnedLibraryBuilder<Op, Type, LA, LD>
where
  Op: Arity
    + Clone
    + Debug
    + Default
    + Ord
    + Sync
    + Send
    + Display
    + std::hash::Hash
    + DiscriminantEq
    + 'static
    + Teachable
    + OperationInfo,
  LA: Debug + Clone + Default,
  LD: Debug + Clone + Default,
  AstNode<Op>: Language + Schedulable<LA, LD>,
  Type: Debug
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
  TypeSet<Type>: ClassMatch,
  AstNode<Op>: TypeInfo<Type>,
{
  #[must_use]
  pub fn learn_trivial(mut self, trivial: bool) -> Self {
    self.learn_trivial = trivial;
    self
  }

  #[must_use]
  pub fn learn_constants(mut self, constants: bool) -> Self {
    self.learn_constants = constants;
    self
  }

  #[must_use]
  pub fn max_arity(mut self, arity: Option<usize>) -> Self {
    self.max_arity = arity;
    self
  }

  #[must_use]
  pub fn ban_op(mut self, op: Op) -> Self {
    self.banned_ops.push(op);
    self
  }

  #[must_use]
  pub fn ban_ops(mut self, iter: impl IntoIterator<Item = Op>) -> Self {
    self.banned_ops.extend(iter);
    self
  }

  #[must_use]
  pub fn with_roots(mut self, roots: Vec<Id>) -> Self {
    self.roots = roots;
    self
  }

  #[must_use]
  pub fn with_co_occurs(mut self, co_occurrences: CoOccurrences) -> Self {
    self.co_occurences = Some(co_occurrences);
    self
  }

  #[must_use]
  pub fn with_last_lib_id(mut self, last_lib_id: usize) -> Self {
    self.last_lib_id = last_lib_id;
    self
  }

  #[must_use]
  pub fn with_clock_period(mut self, clock_period: usize) -> Self {
    self.clock_period = clock_period;
    self
  }

  #[must_use]
  pub fn with_area_estimator(mut self, area_estimator: LA) -> Self {
    self.area_estimator = area_estimator;
    self
  }

  #[must_use]
  pub fn with_delay_estimator(mut self, delay_estimator: LD) -> Self {
    self.delay_estimator = delay_estimator;
    self
  }

  #[must_use]
  pub fn with_bb_query(mut self, bb_query: BBQuery) -> Self {
    self.bb_query = bb_query;
    self
  }

  #[must_use]
  pub fn with_liblearn_config(
    mut self,
    liblearn_config: LiblearnConfig,
  ) -> Self {
    self.liblearn_config = liblearn_config;
    self
  }

  #[must_use]
  pub fn with_meta_au_config(mut self, meta_au_config: MetaAUConfig) -> Self {
    self.meta_au_config = meta_au_config;
    self
  }

  #[must_use]
  pub fn with_ci_encoding_config(
    mut self,
    ci_encoding_config: CiEncodingConfig,
  ) -> Self {
    self.ci_encoding_config = ci_encoding_config;
    self
  }

  #[must_use]
  pub fn find_packs(mut self) -> Self {
    self.find_packs = true;
    self
  }

  #[must_use]
  pub fn with_find_pack_config(
    mut self,
    find_pack_config: FindPackConfig,
  ) -> Self {
    self.find_pack_config = find_pack_config;
    self
  }

  pub fn build(
    self,
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, Type>>,
    roots: Vec<Id>,
  ) -> LearnedLibrary<Op, (Id, Id), Type, LA, LD>
  where
    AstNode<Op>: Language,
    <AstNode<Op> as Language>::Discriminant: Sync + Send,
    Op: crate::ast_node::Printable + OperationInfo,
  {
    // debug!("Computing co-occurences");
    // let co_occurs = self.co_occurences.unwrap_or_else(|| {
    //   let co_ext = COBuilder::new(egraph, roots);
    //   co_ext.run()
    // });

    debug!("Constructing learned libraries");
    LearnedLibrary::new(
      egraph,
      roots,
      self.learn_trivial,
      self.learn_constants,
      self.max_arity,
      self.banned_ops,
      self.last_lib_id,
      self.clock_period,
      self.area_estimator,
      self.delay_estimator,
      self.bb_query,
      self.liblearn_config,
      self.meta_au_config,
      self.ci_encoding_config,
      self.find_packs,
      self.find_pack_config,
    )
  }
}

#[derive(Debug, Clone)]
pub struct LiblearnMessage<Op, Type, A: Analysis<AstNode<Op>>>
where
  Op: Arity
    + Clone
    + Debug
    + Ord
    + Sync
    + Send
    + Display
    + Hash
    + DiscriminantEq
    + 'static
    + Teachable
    + OperationInfo,
  Type: Debug
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
  A: 'static,
{
  pub lib_id: usize,
  pub rewrite: Rewrite<AstNode<Op>, A>,
  pub searcher_pe: PartialExpr<Op, Var>,
  pub applier_pe: PartialExpr<Op, Var>,
  pub condition: TypeMatch<Type>,
  pub search_len: usize,
}

pub trait DiscriminantEq {
  fn discriminant_eq(&self, other: &Self) -> bool;
}

#[derive(Debug, Clone)]
pub struct AUWithType<Op: OperationInfo + Clone + Ord, Type> {
  pub au: AU<Op, Var, Type>,
  pub ty_map: HashMap<Var, Vec<Type>>,
  pub detailed_io: HashMap<String, Type>,
}
impl<Op: PartialEq + Eq + OperationInfo + Clone + Ord, Type> PartialEq
  for AUWithType<Op, Type>
{
  fn eq(&self, other: &Self) -> bool {
    self.au == other.au
  }
}
impl<Op: Eq + OperationInfo + Clone + Ord, Type> Eq for AUWithType<Op, Type> {}

impl<Op: Eq + OperationInfo + Clone + Ord, Type> PartialOrd
  for AUWithType<Op, Type>
{
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    Some(self.au.cmp(&other.au))
  }
}
impl<Op: Eq + OperationInfo + Clone + Ord, Type> Ord for AUWithType<Op, Type> {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.au.cmp(&other.au)
  }
}

/// A `LearnedLibrary<Op>` is a collection of functions learned from an
/// [`EGraph<AstNode<Op>, _>`] by antiunifying pairs of enodes to find their
/// common structure.
///
/// You can create a `LearnedLibrary` using
/// [`LearnedLibrary::from(&your_egraph)`].
#[derive(Debug, Clone)]
pub struct LearnedLibrary<Op, T, Type, LA, LD>
where
  Op: Arity
    + Clone
    + Debug
    + Ord
    + Sync
    + Send
    + Display
    + DiscriminantEq
    + Hash
    + 'static
    + Teachable
    + OperationInfo,
  LA: Debug + Clone,
  LD: Debug + Clone,
  Type: Debug
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
  TypeSet<Type>: ClassMatch,
  AstNode<Op>: TypeInfo<Type>,
  T: Clone + Ord,
{
  egraph: EGraph<AstNode<Op>, ISAXAnalysis<Op, Type>>,
  /// A map from DFTA states (i.e. pairs of enodes) to their antiunifications.
  aus_by_state: BTreeMap<T, BTreeSet<AU<Op, T, Type>>>,
  /// A set of all the antiunifications discovered.
  aus: BTreeSet<AUWithType<Op, Type>>,
  /// Whether to learn "trivial" anti-unifications.
  learn_trivial: bool,
  /// Whether to also learn "library functions" which take no arguments.
  learn_constants: bool,
  /// Maximum arity of functions to learn.
  max_arity: Option<usize>,
  /// Operations that must never appear in learned abstractions.
  banned_ops: Vec<Op>,
  // /// 存储deduplicate_from_candidates中cache已经存储过的值
  // pattern_cache: HashMap<PartialExpr<Op, (Id, Id)>, Vec<Match>>,
  last_lib_id: usize,
  /// clock period used in scheduling
  clock_period: usize,
  /// area estimator
  area_estimator: LA,
  /// delay estimator
  delay_estimator: LD,
  /// BB query for CPU latency
  bb_query: BBQuery,
  /// config for liblearn
  liblearn_config: LiblearnConfig,
  /// meta_au_config
  meta_au_config: MetaAUConfig,
  /// IO filter config
  ci_encoding_config: CiEncodingConfig,
  /// vectorized library
  find_packs: bool,
  /// 包搜索模式的配置
  find_pack_config: FindPackConfig,
}

#[allow(unused)]
fn mem_usage_of<F, X, L>(label: L, f: F) -> (X, usize)
where
  F: FnOnce() -> X,
  L: std::fmt::Display,
{
  let before_mem = memory_stats::memory_stats().unwrap().physical_mem;
  let x = f();
  let after_mem = memory_stats::memory_stats().unwrap().physical_mem;
  // println!("{label} {}mb", ((after_mem - before_mem) as f64) / 1e6);
  (x, after_mem - before_mem)
}

impl<Op, Type, LA, LD> LearnedLibrary<Op, (Id, Id), Type, LA, LD>
where
  Op: Arity
    + Clone
    + Debug
    + Default
    + Ord
    + Sync
    + Send
    + Display
    + DiscriminantEq
    + std::hash::Hash
    + Teachable
    + 'static
    + OperationInfo,
  LA: Debug + Clone,
  LD: Debug + Clone,
  AstNode<Op>: Language + Schedulable<LA, LD>,
  Type: Debug
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
  TypeSet<Type>: ClassMatch,
  AstNode<Op>: TypeInfo<Type>,
{
  fn calculate_all_hash(
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, Type>>,
    ids: Vec<Id>,
    use_meta_au_relaxed_hash: bool,
  ) -> HashMap<Id, StructuralHash> {
    let mut visited = HashSet::new();
    let mut ecls_hash = HashMap::new();
    for id in ids {
      Self::calculate_all_hash_with_visited(
        egraph,
        &mut ecls_hash,
        id,
        &mut visited,
        use_meta_au_relaxed_hash,
      );
    }
    // 如果还存在没有访问的eclass，则计算它们的哈希
    for eclass in egraph.classes() {
      if !visited.contains(&eclass.id) {
        Self::calculate_all_hash_with_visited(
          egraph,
          &mut ecls_hash,
          eclass.id,
          &mut visited,
          use_meta_au_relaxed_hash,
        );
      }
    }
    ecls_hash
  }

  fn calculate_all_hash_with_visited(
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, Type>>,
    ecls_hash: &mut HashMap<Id, StructuralHash>,
    id: Id,
    visited: &mut HashSet<Id>,
    use_meta_au_relaxed_hash: bool,
  ) {
    if visited.contains(&id) {
      return;
    }
    visited.insert(id);

    let eclass = &egraph[id];
    let nodes = eclass.nodes.clone();

    // Recalculate the data for this eclass
    let make_new_structural_hash =
      |node: &AstNode<Op>,
       ecls_hash: &mut HashMap<Id, StructuralHash>|
       -> StructuralHash {
        let compute_hash_level_for_node = |node: &AstNode<Op>, child_hashes: &[u64]| {
          if !use_meta_au_relaxed_hash {
            return compute_hash_level(node, child_hashes);
          }
          let mut hasher = DefaultHasher::new();
          if let Some(key) = node.operation().meta_au_hash_key() {
            key.hash(&mut hasher);
          } else {
            node.operation().to_shielding_op().hash(&mut hasher);
          }
          for &h in child_hashes {
            h.hash(&mut hasher);
          }
          (hasher.finish() % 64) as usize
        };
        let compute_full_hash_for_node = |node: &AstNode<Op>, child_hashes: &[u64]| {
          if !use_meta_au_relaxed_hash {
            return compute_full_hash(node, child_hashes);
          }
          let mut hasher = DefaultHasher::new();
          if let Some(key) = node.operation().meta_au_hash_key() {
            key.hash(&mut hasher);
          } else {
            node.operation().to_shielding_op().hash(&mut hasher);
          }
          for &h in child_hashes {
            hasher.write_u64(h);
          }
          hasher.finish()
        };
        let mut children = node.children().to_vec();
        // 如果有子节点和当前节点相同，则不需要计算哈希
        children.retain(|&child| child != id);
        // 按照operation进行排序，如果operation可交换
        if node.operation().is_commutative() {
          children.sort_by_key(|&child| {
            ecls_hash
              .get(&child)
              .map_or_else(|| StructuralHash::default(), |hash| hash.clone())
              .cls_hash
          });
        }
        // 计算子节点哈希
        let child_hashes = children
          .iter()
          .map(|&child| {
            let hash = ecls_hash
              .get(&child)
              .map_or_else(|| StructuralHash::default(), |hash| hash.clone());
            hash.cls_hash.clone()
          })
          .collect::<Vec<_>>();
        // 计算当前节点哈希
        let current_level = compute_hash_level_for_node(node, &child_hashes);
        // 合并子树层级
        let mut subtree_levels = bitvec![u64, Lsb0; 0; 64];
        subtree_levels.set(current_level, true);
        for child in children.iter() {
          let child_level = ecls_hash.get(child).map_or_else(
            || bitvec![u64, Lsb0; 0; 64],
            |hash| hash.subtree_levels.clone(),
          );
          subtree_levels |= child_level;
        }
        // 计算完整哈希
        let extract_hash = compute_full_hash_for_node(node, &child_hashes);

        let hash = StructuralHash {
          cls_hash: extract_hash,
          subtree_levels,
          level_conflict: LevelConflictState::new(),
        };
        hash
      };
    let mut struc_hashes = Vec::new();
    for (_i, node) in nodes.iter().enumerate() {
      for arg in node.args() {
        if arg == &id {
          // Avoid infinite recursion
          continue;
        }
        Self::calculate_all_hash_with_visited(
          egraph,
          ecls_hash,
          *arg,
          visited,
          use_meta_au_relaxed_hash,
        );
      }
      struc_hashes.push(make_new_structural_hash(node, ecls_hash));
    }
    // 排序
    struc_hashes.sort_by_key(|hash| hash.cls_hash.clone());
    // 合并所有的哈希
    let mut new_data = StructuralHash::default();
    for hash in struc_hashes {
      StructuralHash::merge(&mut new_data, &hash);
    }
    ecls_hash.insert(id, new_data);
  }
  /// Constructs a [`LearnedLibrary`] from an [`EGraph`] by antiunifying pairs
  /// of enodes to find their common structure.
  fn new(
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, Type>>,
    roots: Vec<Id>,
    learn_trivial: bool,
    learn_constants: bool,
    max_arity: Option<usize>,
    banned_ops: Vec<Op>,
    last_lib_id: usize,
    clock_period: usize,
    area_estimator: LA,
    delay_estimator: LD,
    bb_query: BBQuery,
    liblearn_config: LiblearnConfig,
    meta_au_config: MetaAUConfig,
    ci_encoding_config: CiEncodingConfig,
    find_packs: bool,
    find_pack_config: FindPackConfig,
  ) -> Self
  where
    Op: crate::ast_node::Printable,
    <AstNode<Op> as Language>::Discriminant: Sync + Send,
  {
    // 计算所有eclass的哈希
    let ecls_hash =
      Self::calculate_all_hash(egraph, roots, meta_au_config.enable_meta_au);
    for ecls in egraph.classes() {
      // 看看有没有eclass没有被计算哈希
      if !ecls_hash.contains_key(&ecls.id) {
        panic!(
          "Eclass {} has no structural hash, this is a bug in liblearn",
          ecls.id
        );
      }
    }
    let mut learned_lib = Self {
      egraph: egraph.clone(),
      aus_by_state: BTreeMap::new(),
      aus: BTreeSet::new(),
      learn_trivial,
      learn_constants,
      max_arity,
      banned_ops,
      last_lib_id,
      clock_period,
      area_estimator,
      delay_estimator,
      bb_query,
      liblearn_config: liblearn_config.clone(),
      meta_au_config,
      ci_encoding_config,
      find_packs,
      find_pack_config,
    };
    let classes: Vec<_> = egraph.classes().map(|cls| cls.id).collect();

    fn hamming_distance(a: u64, b: u64) -> u32 {
      (a ^ b).count_ones()
    }

    fn jaccard_similarity(a: &u64, b: &u64) -> f64 {
      let intersection = (a.clone() & b.clone()).count_ones() as f64;
      let union = (a.clone() | b.clone()).count_ones() as f64;
      if union == 0.0 {
        1.0 // 完全相同
      } else {
        intersection / union
      }
    }

    fn type_match(a: &String, b: &String) -> bool {
      // FIXME: 判断类型的时候，如果有一方是unknown，那么按理来说应该也能匹配
      // if a == "unknown" || b == "unknown" {
      //   return true;
      // }
      a == b
    }

    fn level_match(
      a: &(u64, u64),
      b: &(u64, u64),
      hamming_threshold: usize,
      jaccard_threshold: f64,
    ) -> bool {
      let hash_similar = hamming_distance(a.0, b.0) < hamming_threshold as u32;
      let subtree_similar = jaccard_similarity(&a.1, &b.1) > jaccard_threshold;
      hash_similar && subtree_similar
    }

    fn group_by_type_ranges(
      class_data: &[(&Id, String, u64, u64)],
    ) -> Vec<(usize, usize)> {
      let mut ranges = vec![];
      let mut i = 0;
      while i < class_data.len() {
        let current_ty = &class_data[i].1;
        let ty_end = class_data[i..]
          .iter()
          .position(|(_, ty, _, _)| ty != current_ty)
          .map(|pos| i + pos)
          .unwrap_or(class_data.len());
        ranges.push((i, ty_end));
        i = ty_end;
      }
      ranges
    }
    match liblearn_config.enum_mode {
      EnumMode::All => {
        // 计算所有pair
        let eclass_pairs = classes
          .iter()
          .cartesian_product(classes.iter())
          .map(|(ecls1, ecls2)| (egraph.find(*ecls1), egraph.find(*ecls2)))
          .collect::<Vec<_>>();
        if meta_au_config.enable_meta_au {
          for pair in eclass_pairs.clone() {
            learned_lib.enumerate_over_egraph_meta_au(egraph, pair);
          }
        } else {
          for pair in eclass_pairs.clone() {
            learned_lib.enumerate_over_egraph(egraph, pair);
          }
        }
      }
      EnumMode::PruningVanilla => {
        // 无剪枝优化+并行处理+模块分组，仅有针对pair的筛选
        let mut eclass_pairs = vec![];
        let mut class_data: Vec<_> = classes
          .iter()
          .map(|cls| {
            let ty = egraph[*cls].data.get_type()[0].clone();
            let structural_hash =
              ecls_hash.get(cls).cloned().unwrap_or_default();
            let cls_hash = structural_hash.cls_hash;
            let subtree_levels = structural_hash.subtree_levels.load();
            (cls, ty, cls_hash, subtree_levels)
          })
          .collect();
        class_data.sort_unstable_by_key(|(ecls, _, _, _)| usize::from(**ecls));
        // 用一个二维数组来存储pairs的配对情况
        for i in 0..class_data.len() {
          for j in i..class_data.len() {
            let (ecls1, ty1, cls_hash1, subtree_levels1) =
              class_data[i].clone();
            let (ecls2, ty2, cls_hash2, subtree_levels2) =
              class_data[j].clone();
            if !level_match(
              &(cls_hash1, subtree_levels1),
              &(cls_hash2, subtree_levels2),
              liblearn_config.hamming_threshold,
              liblearn_config.jaccard_threshold,
            ) {
              continue;
            }
            if !type_match(&ty1, &ty2) {
              continue;
            }
            eclass_pairs.push((*ecls1, *ecls2));
          }
        }
      }
      EnumMode::PruningGold => {
        // 剪枝优化+并行处理
        let mut class_data: Vec<_> = classes
          .iter()
          .map(|cls| {
            let ty = egraph[*cls].data.get_type()[0].clone();
            let structural_hash =
              ecls_hash.get(cls).cloned().unwrap_or_default();
            let cls_hash = structural_hash.cls_hash;
            let subtree_levels = structural_hash.subtree_levels.load();
            (cls, ty, cls_hash, subtree_levels)
          })
          .collect();
        class_data.sort_unstable_by_key(|(ecls, ty, _, _)| {
          (ty.clone(), usize::from(**ecls))
        });
        // for data in class_data.clone() {
        //   println!("Id: {}, Type: {}", data.0, data.1);
        // }
        let ranges = group_by_type_ranges(&class_data);
        // 用一个二维数组来存储pairs的配对情况
        let enum_start = Instant::now();
        let mut op_hash = Vec::new();
        for ecls in egraph.classes() {
          let mut enodes =
            ecls.nodes.iter().map(|x| x.clone()).collect::<Vec<_>>();
          enodes.sort_unstable_by_key(|x| x.operation().clone());
          let ops = enodes
            .iter()
            .map(|x| x.operation().clone())
            .collect::<Vec<_>>();
          op_hash.push((
            enodes,
            ops,
            ecls_hash.get(&ecls.id).cloned().unwrap_or_default(),
            ecls.id,
          ));
        }
        op_hash.sort_unstable_by_key(|(_, ops, _, _)| ops.clone());
        // for (enodes, _, structural_hash, id) in op_hash {
        //   println!(
        //     "{}: enodes: {:?}, StructuralHash: {}",
        //     id, enodes, structural_hash.cls_hash
        //   );
        // }
        let all_pairs: Vec<(Id, Id)> = ranges
          .into_par_iter()
          .flat_map(|(start, end)| {
            let mut local_pairs = vec![];

            for i in start..end {
              let (ecls1, _, cls_hash1, subtree_levels1) = &class_data[i];
              let popcount1 = cls_hash1.count_ones();
              let subtree_cnt1 = subtree_levels1.count_ones();
              for j in i..end {
                // FIXME:ENUM模式
                // if i != j {
                //   continue; // 只考虑同一类型的eclass
                // }
                let (ecls2, _, cls_hash2, subtree_levels2) = &class_data[j];
                // 包搜索模式下，只使用类型匹配
                if find_packs && !find_pack_config.prune_similar {
                  // 不进行后面的检查，直接插入
                  local_pairs.push((**ecls1, **ecls2));
                  continue;
                }
                let popcount2 = cls_hash2.count_ones();
                if (popcount1 as i32 - popcount2 as i32).abs()
                  >= liblearn_config.hamming_threshold as i32
                {
                  continue; // 汉明距离不可能<36
                }
                let subtree_cnt2 = subtree_levels2.count_ones();
                let all_one = (subtree_levels1.clone()
                  | subtree_levels2.clone())
                .count_ones();
                if (subtree_cnt1.max(subtree_cnt2) as f64)
                  < (liblearn_config.jaccard_threshold * all_one as f64)
                {
                  continue; // Jaccard相似度不可能>
                            // liblearn_config.jaccard_threshold
                }
                if !level_match(
                  &(*cls_hash1, subtree_levels1.clone()),
                  &(*cls_hash2, subtree_levels2.clone()),
                  liblearn_config.hamming_threshold,
                  liblearn_config.jaccard_threshold,
                ) {
                  continue;
                }

                local_pairs.push((**ecls1, **ecls2));
              }
            }

            local_pairs
          })
          .collect();
        let eclass_pairs = all_pairs.clone();
        // for pair in eclass_pairs.clone() {
        //   println!(
        //     "eclass_pair: {:?}, enode_op_pair: {:?}, {:?}",
        //     pair,
        //     egraph[pair.0].nodes.iter().map(|x| x.operation()).join(","),
        //     egraph[pair.1].nodes.iter().map(|x| x.operation()).join(","),
        //   );
        // }
        let start = Instant::now();
        if meta_au_config.enable_meta_au {
          for pair in eclass_pairs.clone() {
            learned_lib.enumerate_over_egraph_meta_au(egraph, pair);
          }
        } else {
          for pair in eclass_pairs.clone() {
            learned_lib.enumerate_over_egraph(egraph, pair);
          }
        }
        let elapsed = start.elapsed();
      }
      EnumMode::ClusterTest => {
        let mut eclass_pairs = vec![];
        let prune_start = Instant::now();
        let mut class_data: Vec<_> = classes
          .iter()
          .map(|cls| {
            let ty = egraph[*cls].data.get_type()[0].clone();
            let structural_hash =
              ecls_hash.get(cls).cloned().unwrap_or_default();
            let cls_hash = structural_hash.cls_hash;
            let subtree_levels = structural_hash.subtree_levels.load();
            (cls, ty, cls_hash, subtree_levels)
          })
          .collect();
        class_data.sort_unstable_by_key(|(ecls, _, _, _)| usize::from(**ecls));
        let mut matched_patterns: HashMap<BitVec<u64>, Vec<(usize, usize)>> =
          HashMap::new();
        let mut patterns = vec![];
        for i in 0..class_data.len() {
          for j in i..class_data.len() {
            let (ecls1, ty1, cls_hash1, subtree_levels1) =
              class_data[i].clone();
            let (ecls2, ty2, cls_hash2, subtree_levels2) =
              class_data[j].clone();
            if !type_match(&ty1, &ty2) {
              continue;
            }
            if !level_match(
              &(cls_hash1, subtree_levels1),
              &(cls_hash2, subtree_levels2),
              liblearn_config.hamming_threshold,
              liblearn_config.jaccard_threshold,
            ) {
              continue;
            }
            let pattern = ecls_hash.get(ecls1).unwrap().subtree_levels.clone()
              & ecls_hash.get(ecls2).unwrap().subtree_levels.clone();
            if pattern.count_ones() == 0 {
              continue;
            }
            if matched_patterns.contains_key(&pattern) {
              matched_patterns.get_mut(&pattern).unwrap().push((i, j));
            } else {
              matched_patterns.insert(pattern.clone(), vec![(i, j)]);
              patterns.push(pattern.clone());
            }
          }
        }

        patterns.sort_by(|a, b| b.count_ones().cmp(&a.count_ones()));
        let m = 30;
        for i in 0..patterns.len() {
          if i > m {
            break;
          }
          let pattern = &patterns[i];
          let mut pairs = matched_patterns.get(pattern).unwrap().clone();
          pairs.sort_by(|a, b| b.cmp(a));
          // 如果小于k个pair就直接插入，否则取前k个
          let k = 10;
          let id_pair = pairs
            .clone()
            .into_iter()
            .map(|(i, j)| (classes[i], classes[j]))
            .collect::<Vec<_>>();
          if pairs.len() < k {
            eclass_pairs.extend(id_pair);
          } else {
            // 取前两个
            for i in 0..k {
              eclass_pairs.push(id_pair[i]);
            }
          }

          // println!("pattern: {:?}", pattern);
          // println!("matched patterns: {:?}",
          // matched_patterns.get(pattern));
        }
        println!(
          "         candidate eclass pairs after pruning/filtering: {}",
          eclass_pairs.len()
        );
        let enum_start = Instant::now();

        if meta_au_config.enable_meta_au {
          for pair in eclass_pairs.clone() {
            learned_lib.enumerate_over_egraph_meta_au(egraph, pair);
          }
        } else {
          for pair in eclass_pairs.clone() {
            learned_lib.enumerate_over_egraph(egraph, pair);
          }
        }
      }
    };
    println!(
      "         we all need to calculate {} pairs of eclasses",
      learned_lib.aus_by_state.len()
    );

    // 如果learned_lib中的aus数量大于500，就从排序结果中随机选取500个
    // if learned_lib.aus.len() > 500 {
    //   let mut aus = learned_lib.aus.iter().cloned().collect::<Vec<_>>();
    //   aus.shuffle(&mut rand::thread_rng());
    //   let aus = aus.into_iter().take(500).collect::<BTreeSet<_>>();
    //   learned_lib.aus = aus;
    // }

    // for (state, aus) in &learned_lib.aus_by_state {
    //   println!("state{:?}: {:?}", state, aus);
    // }
    if meta_au_config.enable_meta_au {
      // 如果是meta_au_search，首先将aus分成两部分，一部分Opmask为1，
      // 另外一部分为0
      let mut has_mask_aus = BTreeSet::new();
      let mut no_mast_aus = BTreeSet::new();
      let mut one_mask_count = 0usize;
      let mut multi_mask_count = 0usize;
      let mut multi_mask_examples = Vec::new();
      for au in learned_lib.aus.iter() {
        let profile = meta_au_mask_profile(&au.au.expr);
        let mask_cnt = profile.opmask_count;
        if mask_cnt == 1 {
          one_mask_count += 1;
          has_mask_aus.insert(au.clone());
        } else if mask_cnt == 0 {
          no_mast_aus.insert(au.clone());
        } else {
          multi_mask_count += 1;
          has_mask_aus.insert(au.clone());
          if multi_mask_examples.len() < 3 {
            multi_mask_examples
              .push(format!("{}", Pattern::from(au.au.expr.clone())));
          }
        }
      }
      println!(
        "        • meta AU final mask summary: one_mask={}, zero_mask={}, multi_mask={}",
        one_mask_count,
        no_mast_aus.len(),
        multi_mask_count
      );
      if !multi_mask_examples.is_empty() {
        println!(
          "          final multi-opmask examples: {}",
          multi_mask_examples.join(" || ")
        );
      }
      // 如果has_mask_aus和no_mask_aus中有一个数量大于50，就选择前50个
      let mut sampled_aus = BTreeSet::new();
      if has_mask_aus.len() > meta_au_config.num_meta_au_mask {
        let aus = has_mask_aus.iter().collect::<Vec<_>>();
        let step = aus.len() / meta_au_config.num_meta_au_mask;
        for i in (0..aus.len()).step_by(step) {
          sampled_aus.insert(aus[i].clone());
        }
      } else {
        sampled_aus.extend(has_mask_aus);
      }
      learned_lib.aus = sampled_aus;
    }
    // // 打印一下au_by_state
    // for (state, aus) in &learned_lib.aus_by_state {
    //   println!("State {:?}: {:?}", state, aus);
    // }
    // for au in learned_lib.aus.iter() {
    //   println!("Learned AU: {}", Pattern::from(au.au.expr.clone()));
    // }
    learned_lib
  }
}

impl<Op, T, Type, LA, LD> LearnedLibrary<Op, T, Type, LA, LD>
where
  Op: Arity
    + Clone
    + Debug
    + Default
    + Display
    + Ord
    + Send
    + Sync
    + Teachable
    + OperationInfo
    + DiscriminantEq
    + 'static
    + Hash
    + OperationInfo,
  LA: Debug + Clone,
  LD: Debug + Clone,
  AstNode<Op>: Language + Schedulable<LA, LD>,
  Type: Debug
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
  TypeSet<Type>: ClassMatch,
  AstNode<Op>: TypeInfo<Type>,
  T: Clone + Ord,
{
  /// Get the maximum bitwidth of the lib in a saturated egraph
  fn get_max_bitwidth(
    egraph: EGraph<AstNode<Op>, ISAXAnalysis<Op, Type>>,
  ) -> HashMap<LibId, usize> {
    let mut max_bitwidths = HashMap::new();
    for eclass in egraph.classes() {
      for node in eclass.iter() {
        match node.as_binding_expr() {
          Some(BindingExpr::Lib(id, _, b, _, _, _)) => {
            for app_node in egraph[*b].iter() {
              for var_cls in app_node.args() {
                for var_node in egraph[*var_cls].iter() {
                  for const_cls in var_node.args() {
                    for const_node in egraph[*const_cls].iter() {
                      let bitwidth = const_node.operation().get_bitwidth();
                      if let Some(max_bitwidth) = max_bitwidths.get_mut(&id) {
                        if *max_bitwidth < bitwidth {
                          *max_bitwidth = bitwidth;
                        }
                      } else {
                        max_bitwidths.insert(id, bitwidth);
                      }
                    }
                  }
                }
              }
            }
          }
          _ => {}
        }
      }
    }
    max_bitwidths
  }

  fn make_cost(
    &self,
    searcher: Pattern<AstNode<Op>>,
    applier: Pattern<AstNode<Op>>,
    max_bitwidth: usize,
  ) -> Pattern<AstNode<Op>>
  where
    Op: Teachable + OperationInfo + Default + Arity + Debug + Clone + Eq + Hash,
    AstNode<Op>: Schedulable<LA, LD>,
  {
    let ast = &searcher.ast;
    let new_expr = ast
      .iter()
      .map(|node| match node {
        egg::ENodeOrVar::ENode(ast_node) => {
          let mut new_node = (*ast_node).clone();
          new_node.operation_mut().make_bitwidth(max_bitwidth);
          new_node
        }
        egg::ENodeOrVar::Var(_) => Op::var(0),
      })
      .collect::<Vec<AstNode<Op>>>();

    let rec_expr: RecExpr<AstNode<Op>> = new_expr.into();
    let scheduler = Scheduler::new(
      self.clock_period,
      self.area_estimator.clone(),
      self.delay_estimator.clone(),
      self.bb_query.clone(),
    );
    let (lat_cpu, lat_acc, area) = scheduler.asap_schedule(&rec_expr);
    let mut new_applier = applier.clone();
    // println!(
    //   "applier: {}, lat_acc: {}, lat_cpu: {}, area: {}",
    //   applier, lat_acc, lat_cpu, area
    // );
    for node in new_applier.ast.iter_mut() {
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
    new_applier
  }

  fn genericize_pe(&self, pe: PartialExpr<Op, Var>) -> PartialExpr<Op, Var> {
    match pe {
      PartialExpr::Hole(_) => pe,
      PartialExpr::Node(node) => {
        let mut new_node = node.clone();
        new_node.operation_mut().genericize();
        for arg in new_node.args_mut() {
          *arg = self.genericize_pe(arg.clone());
        }
        PartialExpr::Node(new_node)
      }
    }
  }

  /// Returns an iterator over rewrite rules that replace expressions with
  /// equivalent calls to a learned library function.
  ///
  /// For example, the expression
  ///
  /// ```text
  /// (* (+ 1 2) 5)
  /// ```
  ///
  /// might be rewritten to
  ///
  /// ```text
  /// (lib f (lambda (* (+ 1 $0) 5))
  ///  (apply f 2))
  /// ```
  ///
  /// by a rewrite rule
  ///
  /// ```text
  /// (* (+ 1 ?x) 5) => (lib f (lambda (* (+ 1 $0) 5)) (apply f ?x))
  /// ```
  pub fn messages(
    &self,
  ) -> Vec<LiblearnMessage<Op, Type, ISAXAnalysis<Op, Type>>> {
    // 直接使用aus会导致问题，因为类似arg这样的节点会带入多余信息，
    // 实际上使用的是同一个库，
    // 所以需要一个HashMap记录libid和对应的去除了多余信息的au
    let mut au_map_to_libid: HashMap<
      PartialExpr<Op, Var>,
      (usize, HashMap<Var, Vec<Type>>, usize),
    > = HashMap::new();
    // 双向映射
    let mut libid_to_generic_au: HashMap<usize, PartialExpr<Op, Var>> =
      HashMap::new();
    let mut generic_au_to_libid: HashMap<PartialExpr<Op, Var>, usize> =
      HashMap::new();
    let mut new_lib_id = self.last_lib_id;
    for au in self.aus.iter() {
      let genericized_pe = self.genericize_pe(au.au.expr.clone());
      if let Some(lib_id) = generic_au_to_libid.get(&genericized_pe) {
        // 如果已经存在这个库，就不需要再添加了
        // 只将库id加入对应的au
        au_map_to_libid.insert(
          au.au.expr.clone(),
          (lib_id.clone(), au.ty_map.clone(), au.au.search_len),
        );
      } else {
        // 如果不存在这个库，就添加一个新的库
        au_map_to_libid.insert(
          au.au.expr.clone(),
          (new_lib_id, au.ty_map.clone(), au.au.search_len),
        );
        libid_to_generic_au.insert(new_lib_id, genericized_pe.clone());
        generic_au_to_libid.insert(genericized_pe, new_lib_id);
        new_lib_id += 1;
      }
    }

    // 把lib_id打印出来

    let msgs = au_map_to_libid.iter().enumerate().map(
      |(iter, (au_pe, (i, ty_map, search_len)))| {
        let searcher: Pattern<_> = au_pe.clone().into();
        let applier_pe = reify(
          LibId(*i),
          libid_to_generic_au.get(i).unwrap().clone(),
          self.clock_period,
          self.area_estimator.clone(),
          self.delay_estimator.clone(),
          self.bb_query.clone(),
        );
        let applier: Pattern<_> = applier_pe.clone().into();
        let conditional_applier = ConditionalApplier {
          condition: TypeMatch::new(ty_map.clone()),
          applier: applier.clone(),
        };
        let id = iter + self.last_lib_id;
        let name = if self.meta_au_config.enable_meta_au {
          format!("meta_anti-unify {id}")
        } else {
          format!("anti-unify {id}")
        };
        debug!("Found rewrite \"{name}\":\n{searcher} => {applier}");
        let condition = TypeMatch::new(ty_map.clone());

        // Both patterns contain the same variables, so this can never fail.
        LiblearnMessage {
          lib_id: *i,
          rewrite: Rewrite::new(name, searcher.clone(), conditional_applier)
            .unwrap_or_else(|_| unreachable!()),
          searcher_pe: au_pe.clone(),
          applier_pe: applier_pe.clone(),
          condition,
          search_len: *search_len,
        }
      },
    );
    // 需要先进行去重处理
    let rules = msgs
      .clone()
      .map(|msg| msg.rewrite)
      .collect::<Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, Type>>>>();
    let runner = Runner::<_, _, ()>::new(ISAXAnalysis::default())
      .with_egraph(self.egraph.clone())
      .with_time_limit(Duration::from_secs(1000))
      .with_iter_limit(1)
      .run(&rules);
    let max_bitwidths = Self::get_max_bitwidth(runner.egraph);
    msgs
      .map(|msg| {
        let i = msg.lib_id;
        let max_bitwidth = max_bitwidths.get(&LibId(i)).cloned().unwrap_or(32);
        // 如果max_bitwidth为0，就设置为32
        let max_bitwidth = if max_bitwidth == 0 { 32 } else { max_bitwidth };
        let applier = self.make_cost(
          msg.searcher_pe.clone().into(),
          msg.applier_pe.clone().into(),
          max_bitwidth,
        );
        let searcher_pe = msg.searcher_pe.clone();
        let searcher: Pattern<_> = searcher_pe.clone().into();
        // let conditional_applier = ConditionalApplier {
        //   condition: msg.condition.clone(),
        //   applier: applier.clone(),
        // };
        let name = if self.meta_au_config.enable_meta_au {
          format!("meta_anti-unify {i}")
        } else {
          format!("anti-unify {i}")
        };
        let applier_pe = applier.clone().into();
        LiblearnMessage {
          lib_id: i,
          rewrite: Rewrite::new(name, searcher, applier)
            .unwrap_or_else(|_| unreachable!()),
          searcher_pe,
          applier_pe,
          condition: msg.condition.clone(),
          search_len: msg.search_len,
        }
      })
      .collect::<Vec<LiblearnMessage<Op, Type, ISAXAnalysis<Op, Type>>>>()
  }

  /// Right-hand sides of library rewrites.
  pub fn libs(&self) -> impl Iterator<Item = Pattern<AstNode<Op>>> + '_ {
    self.aus.iter().enumerate().map(|(i, au)| {
      let new_i = i + self.last_lib_id;
      let applier: Pattern<_> = reify(
        LibId(new_i),
        au.au.expr.clone(),
        self.clock_period,
        self.area_estimator.clone(),
        self.delay_estimator.clone(),
        self.bb_query.clone(),
      )
      .into();
      applier
    })
  }

  pub fn for_each_anti_unification<F>(&mut self, f: F)
  where
    F: Fn(&PartialExpr<Op, Var>) -> PartialExpr<Op, Var>,
  {
    let mut new_aus = BTreeSet::new();
    for au in &self.aus {
      let new_au = f(&au.au.expr);
      new_aus.insert(AUWithType {
        au: AU::new(
          new_au.clone(),
          au.au.delay,
          au.au.latency_gain,
          au.au.area_cost,
          self.liblearn_config.cost.clone(),
        ),
        ty_map: au.ty_map.clone(),
        detailed_io: au.detailed_io.clone(),
      });
    }
    self.aus = new_aus;
  }

  /// The raw anti-unifications that we have collected
  pub fn anti_unifications(
    &self,
  ) -> impl Iterator<Item = &PartialExpr<Op, Var>> {
    self.aus.iter().map(|au| &au.au.expr)
  }

  /// Extend the set of anti-unifications externally
  pub fn extend(
    &mut self,
    aus: impl IntoIterator<Item = AUWithType<Op, Type>>,
  ) {
    self.aus.extend(aus);
  }

  /// Number of patterns learned.
  #[must_use]
  pub fn size(&self) -> usize {
    self.aus.len()
  }

  pub fn delete_libs_according_size(&mut self) {
    // 删除小于liblearn_config.min_lib_size的库
    self.aus.retain(|au| {
      au.au.expr.size() >= self.liblearn_config.min_lib_size
        && au.au.expr.size() <= self.liblearn_config.max_lib_size
    });
  }

  /// If two candidate patterns (stored in `nontrivial_aus`) have the same set
  /// of matches, only preserve the smaller one of them.
  /// Here a match is a pair of the e-class where the match was found
  /// and the range of its substitution
  /// (as a multiset of e-classes; which variables matched which e-classes is
  /// irrelevant). The reason two such patterns are equivalent is because
  /// their corresponding library functions can be used in exactly the same
  /// places and will have the same multiset (and hence size) of actual
  /// arguments.
  ///
  /// For example, after running a DSR (+ ?x ?y) => (+ ?y ?x),
  /// for any learned pattern containing (+ ?x0 ?x1), there will be an
  /// equivalent pattern containing (+ ?x1 ?x0), which will be eliminated
  /// here.
  pub fn deduplicate(
    &mut self,
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, Type>>,
  ) where
    <AstNode<Op> as Language>::Discriminant: Sync + Send,
  {
    // println!("before deduplicating: ");
    // for lib in self.libs().collect::<Vec<_>>() {
    //   println!("{}", lib);
    // }
    // The algorithm is simply to iterate over all patterns,
    // and save their matches in a dictionary indexed by the match set.
    let mut cache: BTreeMap<Vec<Match>, AUWithType<Op, Type>> = BTreeMap::new();
    let mut i = 0;
    for au in &self.aus {
      // println!("Processing anti-unification {i}");
      i += 1;
      // println!("expr: {}", Pattern::from(au.au.expr.clone()));
      if au.au.expr.size() < self.liblearn_config.min_lib_size {
        // println!("{}: size filter failed, too small", i);
        continue;
      }
      if au.au.expr.size() > self.liblearn_config.max_lib_size {
        // println!("{}: size filter failed, too large", i);
        continue;
      }
      // 如果latency_gain和area_cost有一项为0，就直接去掉
      if au.au.latency_gain == 0.0 {
        // println!("pattern: {}", Pattern::from(au.au.expr.clone()));
        // println!("{}: latency_gain is 0", i);

        debug!(
          "Pruning pattern {} as it has latency_gain or area_cost of 0",
          Pattern::from(au.au.expr.clone())
        );
        continue;
      }
      // FIXME: 这个需要拉到后面
      // 如果io不满足约束，那么就直接跳过
      // if !io_filter(&self.ci_encoding_config, &au.detailed_io) {
      //   // println!("{}: io filter failed", i);
      //   debug!(
      //     "Pruning pattern {} as it does not satisfy IO constraints",
      //     Pattern::from(au.au.expr.clone())
      //   );
      //   continue;
      // }

      let pattern: Pattern<_> = au.au.expr.clone().into();
      // A key in `cache` is a set of matches
      // represented as a sorted vector.
      let mut key = vec![];
      let (tx, rx) = mpsc::channel();

      // 克隆 pattern 和 egraph，move 到新线程
      let pattern_clone = pattern.clone();
      let egraph_clone = egraph.clone();

      std::thread::spawn(move || {
        // 执行搜索
        // 这里假设 search 返回 Vec<SearchMatches<'_, _>>，并且可以变成 'static
        // 如果 SearchMatches 包含对 egraph 引用的借用，且无法满足 'static，
        // 可能需要先将结果转换成可 Send 的结构（例如复制必要的数据）。
        let results: Vec<(Id, Vec<Subst>)> = pattern_clone
          .search(&egraph_clone)
          .into_iter()
          .map(|m| (m.eclass, m.substs))
          .collect();

        // 如果 SearchMatches 包含对 egraph 的引用，那么要注意生命周期问题——
        // 标准 mpsc 发送的数据需满足 Send +
        // 'static，所以实际使用时可能需要调整类型， 或者仅提取 eclass
        // id 等可传输的数据。 下面假设我们只需要 eclass id
        // Vec，或者已转换成可 Send 的形式。 例如：
        // let ids: Vec<Id> = results.iter().map(|m| m.eclass).collect();
        // tx.send(ids).ok();

        // 如果直接发送 results，确保其类型满足 Send + 'static:
        let _ = tx.send(results);
      });

      // 等待结果或超时
      let results = match rx.recv_timeout(Duration::from_secs(5)) {
        Ok(results) => {
          // 主线程在超时前收到了搜索结果
          // 注意：这里的 results 类型要与发送时匹配
          // 可能需要做生命周期转换，或提前转换成独立数据
          // 例如若发送的是 Vec<Id>，这里就返回 Vec<Id> 等
          // println!(
          //   "Search for pattern {} completed with {} matches, size: {},
          // nodes: {}",   Pattern::from(au.au.expr.clone()),
          //   results.len(),
          //   au.au.expr.size(),
          //   au.au.expr.num_nodes(),
          // );
          Ok(results)
        }
        Err(err) => {
          // println!(
          //   "Search for pattern {} timed out or failed: {:?}, size: {},
          // nodes: {}",   Pattern::from(au.au.expr.clone()),
          //   au.au.expr.size(),
          //   au.au.expr.num_nodes(),
          //   err
          // );
          // 超时或通道关闭等
          // 超时: Err(RecvTimeoutError::Timeout)
          // 通道关闭: Err(RecvTimeoutError::Disconnected)
          Err(())
        }
      };

      // 如果results.len() <= 1，就说明没有被复用，直接跳过即可
      if results.is_err()
        || (results.as_ref().unwrap().len() <= 1 && !au.au.expr.has_vector_op())
      {
        debug!(
          "Pruning pattern {} as it has no matches or search timed out",
          Pattern::from(au.au.expr.clone())
        );
        continue;
      }

      // 首先将结果填充为具体的节点
      let results = results.unwrap();
      // let au = Self::fill_au(au, egraph, &results);

      // 填充后，使用io_filter来过滤
      let detailed_io: Vec<Type> = au.detailed_io.values().cloned().collect();
      if !io_filter(&self.ci_encoding_config, &detailed_io) {
        debug!(
          "Pruning pattern {} as it does not satisfy IO constraints after
      filling",
          Pattern::from(au.au.expr.clone())
        );
        continue;
      }

      for (ecls_id, substs) in results {
        for sub in substs {
          let actuals: Vec<_> =
            pattern.vars().iter().map(|v| sub[*v]).collect();
          let match_signature = Match::new(ecls_id, actuals);
          key.push(match_signature);
        }
      }

      key.sort();
      match cache.get(&key) {
        Some(cached)
          if cached.au.latency_gain >= au.au.latency_gain
            && cached.au.area_cost <= au.au.area_cost =>
        {
          debug!(
            "Pruning pattern {}\n as a duplicate of {}",
            pattern,
            Pattern::from(cached.au.expr.clone())
          );
        }
        _ => {
          cache.insert(key, au.clone());
        }
      }
    }
    let deduplicated_aus: BTreeSet<AUWithType<Op, Type>> =
      cache.into_iter().map(|(_, au)| au).collect();
    // 如果aus的数量大于max_libs，就按照latency_gain从大到小排序，
    // 选择最大的max_libs个
    if deduplicated_aus.len() > self.liblearn_config.max_libs {
      let mut sorted_aus: Vec<_> = deduplicated_aus.into_iter().collect();
      sorted_aus.sort_by(|a, b| {
        OrderedFloat(b.au.latency_gain)
          .cmp(&OrderedFloat(a.au.latency_gain))
          .then(a.au.area_cost.cmp(&b.au.area_cost))
          .then(a.au.expr.size().cmp(&b.au.expr.size()))
      });
      self.aus = sorted_aus
        .into_iter()
        .take(self.liblearn_config.max_libs)
        .collect();
    } else {
      self.aus = deduplicated_aus;
    }
    // for au in &self.aus {
    //   println!(
    //     "latency_gain: {}, area_cost: {}",
    //     au.au.latency_gain, au.au.area_cost
    //   );
    // }
  }
  /// 本函数用来将au中部分var填充为具体的节点
  pub fn fill_au(
    au: &AUWithType<Op, Type>,
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, Type>>,
    results: &Vec<(Id, Vec<Subst>)>,
  ) -> AUWithType<Op, Type> {
    let vars = Pattern::from(au.au.expr.clone()).vars().clone();
    // 将results中的Vec<Subst>展平成一个
    let substs = results
      .iter()
      .flat_map(|(_, subs)| subs.iter())
      .collect::<Vec<_>>();
    // 每个subst是一个SmallVec<Var, Id>，需要收集每个Var对应的Id为一个HashSet
    let mut var_match_map: HashMap<Var, HashSet<Id>> = HashMap::new();
    for sub in substs {
      for var in vars.iter() {
        if let Some(id) = sub.get(*var) {
          var_match_map
            .entry(*var)
            .or_insert_with(HashSet::new)
            .insert(*id);
        }
      }
    }
    // 对于每个Var，如果匹配的Id只有一个，
    // 并且对应在egraph中存在args数目为0的enode，那么取出这个enode
    let mut filled_vars = HashMap::new();
    for (var, ids) in var_match_map.iter() {
      if ids.len() == 1 {
        // 取出这个Id
        let id = *ids.iter().next().unwrap();
        // 在egraph中查找这个Id对应的enode
        let nodes = egraph[id].nodes.clone();
        for node in nodes.iter() {
          // 如果这个enode的args数目为0，就取出这个enode
          if node.args().is_empty() {
            let new_node: AstNode<Op, PartialExpr<Op, Var>> =
              AstNode::new(node.operation().clone(), vec![]);
            filled_vars.insert(*var, new_node);
            break; // 找到后就跳出循环
          }
        }
      }
    }
    let mut fill_holes = |var: Var| -> PartialExpr<Op, Var> {
      if filled_vars.contains_key(&var) {
        // 如果filled_vars中有这个var，就用对应的enode替换
        PartialExpr::Node(filled_vars[&var].clone())
      } else {
        // 否则返回一个Hole
        PartialExpr::Hole(var)
      }
    };
    let filled_expr = au.au.expr.clone().fill(&mut fill_holes);
    let mut new_au = au.clone();
    // 更新search_len
    new_au.au.set_search_len(results.len());
    new_au.au.expr = filled_expr;
    // 除此之外， 还需要更新detailed_io和对应的ty_map，
    // 将替换后的var对应的type对应的项删掉
    filled_vars.keys().for_each(|var| {
      new_au.detailed_io.remove(&var.to_string());
      new_au.ty_map.remove(var);
    });

    new_au
  }

  // 这个函数可以用，只不过将Hole排除在外就可以
  pub fn deduplicate_from_candidates(
    &self,
    candidates: Vec<AUWithType<Op, Type>>, /*
                                           修改为 (Id, Id) */
  ) -> Vec<AUWithType<Op, Type>>
  where
    <AstNode<Op> as Language>::Discriminant: Sync + Send,
  {
    // println!("begin deduplicate_from_candidates");
    // Holes不参与对比和组内去重
    let mut holes = vec![];
    // 修改返回类型
    // 创建一个缓存，用于保存已经遇到的匹配集合与对应的最小模式
    let mut cache: BTreeMap<Vec<Match>, AUWithType<Op, Type>> = BTreeMap::new();
    // info!("cache.size: {}", self.pattern_cache.len());
    // 遍历所有候选的模式
    for au in candidates {
      // 如果au是Hole，加入holes
      if matches!(au.au.expr, PartialExpr::Hole(_)) {
        holes.push(au);
        continue;
      }
      if au.au.expr.size() > self.liblearn_config.max_lib_size {
        continue;
      }
      let pattern: Pattern<_> = au.au.expr.clone().into();
      // A key in `cache` is a set of matches
      // represented as a sorted vector.
      let mut key = vec![];
      let (tx, rx) = mpsc::channel();

      // 克隆 pattern 和 egraph，move 到新线程
      let pattern_clone = pattern.clone();
      let egraph_clone = self.egraph.clone();
      // println!(
      //   "Processing anti-unification: {}",
      //   Pattern::from(au.au.expr.clone())
      // );
      std::thread::spawn(move || {
        // 执行搜索
        // 这里假设 search 返回 Vec<SearchMatches<'_, _>>，并且可以变成 'static
        // 如果 SearchMatches 包含对 egraph 引用的借用，且无法满足 'static，
        // 可能需要先将结果转换成可 Send 的结构（例如复制必要的数据）。
        let results: Vec<(Id, Vec<Subst>)> = pattern_clone
          .search(&egraph_clone)
          .into_iter()
          .map(|m| (m.eclass, m.substs))
          .collect();

        // 如果 SearchMatches 包含对 egraph 的引用，那么要注意生命周期问题——
        // 标准 mpsc 发送的数据需满足 Send +
        // 'static，所以实际使用时可能需要调整类型， 或者仅提取 eclass
        // id 等可传输的数据。 下面假设我们只需要 eclass id
        // Vec，或者已转换成可 Send 的形式。 例如：
        // let ids: Vec<Id> = results.iter().map(|m| m.eclass).collect();
        // tx.send(ids).ok();

        // 如果直接发送 results，确保其类型满足 Send + 'static:
        let _ = tx.send(results);
      });

      // 等待结果或超时
      let results = match rx.recv_timeout(Duration::from_secs(500)) {
        Ok(results) => {
          // 主线程在超时前收到了搜索结果
          // 注意：这里的 results 类型要与发送时匹配
          // 可能需要做生命周期转换，或提前转换成独立数据
          // 例如若发送的是 Vec<Id>，这里就返回 Vec<Id> 等
          Ok(results)
        }
        Err(err) => {
          // 超时或通道关闭等
          // 超时: Err(RecvTimeoutError::Timeout)
          // 通道关闭: Err(RecvTimeoutError::Disconnected)
          Err(())
        }
      };

      if results.is_err() {
        debug!("Due to search timeout or error, I just skip this pattern",);
        continue;
      }

      for (ecls_id, substs) in results.unwrap() {
        for sub in substs {
          let actuals: Vec<_> =
            pattern.vars().iter().map(|v| sub[*v]).collect();
          let match_signature = Match::new(ecls_id, actuals);
          key.push(match_signature);
        }
      }

      key.sort();

      // 如果缓存中已经有相同的匹配集合，则只保留较小的那个
      match cache.get(&key) {
        Some(cached) => {
          //取latency_gain最大的
          if cached.au.latency_gain < au.au.latency_gain {
            cache.insert(key, au);
          } else if cached.au.latency_gain == au.au.latency_gain
            && cached.au.area_cost > au.au.area_cost
          {
            cache.insert(key, au);
          } else if cached.au.latency_gain == au.au.latency_gain
            && cached.au.area_cost == au.au.area_cost
            && cached.au.expr.size() > au.au.expr.size()
          {
            cache.insert(key, au);
          } else if cached.au.expr.clone().hole() > au.au.expr.clone().hole() {
            cache.insert(key, au);
          } else {
            // 如果cached的au比au小，就不做任何操作
            debug!(
              "Pruning pattern {} as a duplicate of {}",
              Pattern::from(au.au.expr.clone()),
              Pattern::from(cached.au.expr.clone())
            );
          }
        }
        None => {
          cache.insert(key, au);
        }
      }
    }
    // 返回缓存中的所有模式,以及所有的holes
    let mut aus = cache.into_iter().map(|(_, au)| au).collect::<Vec<_>>();
    aus.extend(holes);
    // println!("deduplicate_from_candidates finished, size: {}", aus.len());
    aus
  }
}

impl<Op, Type, LA, LD> LearnedLibrary<Op, (Id, Id), Type, LA, LD>
where
  Op: Arity
    + Clone
    + Debug
    + Default
    + Ord
    + DiscriminantEq
    + Hash
    + Sync
    + Send
    + Display
    + 'static
    + Teachable
    + OperationInfo,
  Type: Debug
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
  TypeSet<Type>: ClassMatch,
  LA: Debug + Clone,
  LD: Debug + Clone,
  AstNode<Op>: TypeInfo<Type> + Schedulable<LA, LD>,
{
  // /// Computes the antiunifications of `state` in the DFTA `dfta`.
  // fn enumerate_over_dfta(
  //   &mut self,
  //   dfta: &Dfta<(Op, Op), (Id, Id)>,
  //   state: (Id, Id),
  // ) {
  //   if self.aus_by_state.contains_key(&state) {
  //     // We've already enumerated this state, so there's nothing to do.
  //     return;
  //   }
  //   info!("Enumerating over state {:?}", state);
  //   // We're going to recursively compute the antiunifications of the inputs
  //   // of all of the rules leading to this state. Before we do, we need to
  //   // mark this state as in progress so that a loop in the rules doesn't
  //   // cause infinite recursion.
  //   //
  //   // By initially setting the antiunifications of this state to empty, we
  //   // exclude any antiunifications that would come from looping sequences
  //   // of rules.
  //   self.aus_by_state.insert(state, BTreeSet::new());

  //   if !self.co_occurrences.may_co_occur(state.0, state.1) {
  //     // a在b的co-occurrence中或者b在a的co-occurrence中
  //     return;
  //   }

  //   let mut aus: BTreeSet<AU<Op, (Id, Id), Type>> = BTreeSet::new();

  //   let mut same = false;
  //   let mut different = false;

  //   // if there is a rule that produces this state
  //   if let Some(rules) = dfta.get_by_output(&state) {
  //     // 获得可以产生当前状态的rule(输入状态和op)
  //     for ((op1, op2), inputs) in rules {
  //       info!(
  //         "op1: {:?}, op2: {:?}, input size: {}",
  //         op1,
  //         op2,
  //         inputs.len()
  //       );
  //       if op1 == op2 {
  //         same = true;
  //         if inputs.is_empty() {
  //           let new_au = PartialExpr::from(AstNode::leaf(op1.clone()));
  //           aus.insert(AU::new_with_expr(
  //             new_au,
  //             &self.egraph,
  //             self.liblearn_config.cost.clone(),
  //           ));
  //         } else {
  //           // Recursively enumerate the inputs to this rule.
  //           for &input in inputs {
  //             self.enumerate_over_dfta(dfta, input);
  //           }

  //           // For a rule `op(s1, ..., sn) -> state`, we add an
  //           // antiunification of the form `(op a1 ... an)` for every
  //           // combination `a1, ..., an` of antiunifications of the
  //           // input states `s1, ..., sn`, i.e., for every `(a1, ..., an)`
  //           // in the cartesian product
  //           // `antiunifications_by_state[s1] × ... ×
  //           // antiunifications_by_state[sn]`

  //           let smax_arity = self.max_arity;

  //           let new_aus = {
  //             let au_range = inputs.iter().map(|input| {
  //               let data: Vec<_> =
  //                 self.aus_by_state[input].iter().cloned().collect();
  //               data
  //             });
  //             let au_range: Vec<Vec<AU<Op, (Id, Id), Type>>> =
  //               au_range.collect::<Vec<_>>();
  //             let new_au = match self.liblearn_config.au_merge_mod {
  //               AUMergeMod::Random => get_random_aus(au_range, 1000),
  //               AUMergeMod::Kd => kd_random_aus(au_range, 1000),
  //               AUMergeMod::Greedy => boundary_aus(au_range),
  //               AUMergeMod::Catesian => boundary_aus(au_range),
  //             };
  //             // get_random_aus(au_range, 10)
  //             //beam_search_aus(au_range, 1000, 2000)
  //             // genetic_algorithm_aus(au_range, 1000, 500, 50)
  //             // au_range.into_iter().multi_cartesian_product()
  //             new_au
  //               .into_iter()
  //               .map(|inputs| {
  //                 // println!("inputs length is {}", inputs.len());
  //                 // let inputs = inputs.into_iter().map(|au|
  // au.expr.clone());                 let result =
  //                   PartialExpr::from(AstNode::new(op1.clone(), inputs));
  //                 result
  //               })
  //               .filter(|au| {
  //                 let result = smax_arity.map_or(true, |max_arity| {
  //                   au.unique_holes().len() <= max_arity
  //                 });
  //                 result
  //               })
  //           };
  //           // info!("length: {}",new_aus.clone().count());
  //           // let mut new_aus = new_aus.collect::<Vec<_>>();
  //           // info!("new_aus is done");
  //           // // 如果超过1000个，就随机抽取1000个
  //           // if new_aus.len() > 100 {
  //           //   let mut new_aus_vec: Vec<PartialExpr<Op, (Id, Id)>> =
  //           // new_aus.clone();   let mut rng = rand::thread_rng();
  //           //   new_aus_vec.shuffle(&mut rng);
  //           //   new_aus = new_aus_vec.into_iter().take(100).collect();
  //           // }
  //           // // 使用deduplicate_from_candidates去重
  //           info!("aus_state.len() is {}", self.aus_by_state.len());
  //           info!("deduplicating new_aus last: {} ",
  // new_aus.clone().count());           let new_aus_dedu = self
  //             .deduplicate_from_candidates::<SimpleAnalysis<Op, Type>>(
  //               new_aus.clone(),
  //             );
  //           info!("now  is {}", new_aus_dedu.len());
  //           let start_total = Instant::now(); // 总的执行时间
  //           aus.extend(new_aus_dedu);
  //           info!(
  //             "Total processing time: {:?}, lenth of new_aus and aus is {}",
  //             start_total.elapsed(),
  //             aus.len()
  //           ); // 总的处理时间
  //         }
  //       } else {
  //         different = true;
  //       }
  //     }
  //   }

  //   if same && different {
  //     let new_expr = PartialExpr::Hole(state);
  //     aus.insert(AU::new_with_expr(
  //       new_expr,
  //       &self.egraph,
  //       self.liblearn_config.cost.clone(),
  //     ));
  //   }
  //   self.filter_aus(aus, state);
  // }

  /// meta_au enumerate
  /// no pruning, judge by the length of args
  fn enumerate_over_egraph_meta_au<A: Analysis<AstNode<Op>> + Clone>(
    &mut self,
    egraph: &EGraph<AstNode<Op>, A>,
    state: (Id, Id),
  ) where
    <A as Analysis<AstNode<Op>>>::Data: ClassMatch,
  {
    if self.aus_by_state.contains_key(&state) {
      // we have already enumerated this state
      return;
    }

    self.aus_by_state.insert(state, BTreeSet::new());

    let mut aus: BTreeSet<AU<Op, (Id, Id), Type>> = BTreeSet::new();

    let mut same = false;
    let mut different = false;

    let ops1 = egraph[state.0].nodes.iter().map(AstNode::as_parts);
    let ops2 = egraph[state.1].nodes.iter().map(AstNode::as_parts);

    for (op1, args1) in ops1 {
      for (op2, args2) in ops2.clone() {
        // Meta AU may abstract over the operation itself, but only for
        // compatible operator families. Arity alone is too weak: it would pack
        // unrelated/effectful operations such as add and load.
        if args1.len() == args2.len()
          && (op1 == op2 || op1.meta_au_compatible_with(op2))
        {
          same = true;
          let new_op = if op1 == op2 {
            // if the same, just use the first one
            op1.clone()
          } else {
            // if different, add a mask op
            Op::make_opmask()
          };
          if args1.is_empty() && args2.is_empty() {
            // 一个小改动，叶节点不会插入Opmask，没有意义
            if op1 == op2 && !op1.is_rule_var() {
              let new_au = PartialExpr::from(AstNode::leaf(op1.clone()));
              aus.insert(AU::new_with_expr(
                new_au,
                self.liblearn_config.cost.clone(),
              ));
            } else {
              different = true;
            }
          } else {
            info!("Processing op1 {:?} and op2 {:?}", op1, op2);
            // recursively enumerate the inputs to this rule.
            let inputs: Vec<_> =
              args1.iter().copied().zip(args2.iter().copied()).collect();

            for next_state in &inputs {
              self.enumerate_over_egraph_meta_au(egraph, *next_state);
            }

            let smax_arity = self.max_arity;
            // let au_range: Vec<Vec<PartialExpr<Op, (Id, Id)>>> =
            // au_range.collect::<Vec<_>>();
            let new_aus = inputs.iter().map(|input| {
              let aus =
                self.aus_by_state[input].iter().cloned().collect::<Vec<_>>();
              aus
            });
            let au_range = new_aus.collect::<Vec<_>>();
            let scheduler = Scheduler::new(
              self.clock_period,
              self.area_estimator.clone(),
              self.delay_estimator.clone(),
              self.bb_query.clone(),
            );
            let au_merger = AUMerger::new(
              new_op,
              scheduler,
              self.liblearn_config.au_merge_mod,
              self.liblearn_config.cost,
              self.liblearn_config.sample_num,
              10,
            );
            let new_aus = au_merger
              .merge(au_range)
              .into_iter()
              .filter(|au| {
                smax_arity.map_or(true, |max_arity| {
                  au.unique_holes().len() <= max_arity
                })
              })
              .filter(|au| {
                // 过滤掉过大的
                let size = au.size();
                size <= self.liblearn_config.max_lib_size
              });

            info!("aus_state.len() is {}", self.aus_by_state.len());
            // info!("deduplicating new_aus last: {} ",
            // new_aus.clone().count());

            // let new_aus_dedu = self
            //   .deduplicate_from_candidates::<SimpleAnalysis<Op,
            // Type>>(new_aus); info!("now  is {}",
            // new_aus_dedu.len());
            let start_total = Instant::now(); // 总的执行时间
                                              // 为每一个新的模式生成一个AU
            aus.extend(new_aus.map(|au| {
              AU::new_with_expr(au, self.liblearn_config.cost.clone())
            }));
            // println!("aus_size: {}", aus.len());
            // println!("{:?}", new_aus_dedu);
            // aus.extend(new_aus_dedu);
            info!(
              "Total processing time: {:?}, lenth of new_aus and aus is {}",
              start_total.elapsed(),
              aus.len()
            ); // 总的处理时间
          }
        } else {
          different = true;
        }
      }
    }

    if same && different {
      let new_expr = PartialExpr::Hole(state);
      aus.insert(AU::new_with_expr(
        new_expr,
        self.liblearn_config.cost.clone(),
      ));
    }

    self.filter_aus(aus, state, egraph);
  }

  /// Computes the antiunifications of `state` in the `egraph`.
  /// This avoids computing all of the cross products ahead of time, which is
  /// what the `Dfta` implementation does. So it should be faster and more
  /// memory-efficient. However, I'm not totally sure it's equivalent to the
  /// `Dfta` version.
  fn enumerate_over_egraph<A: Analysis<AstNode<Op>> + Clone>(
    &mut self,
    egraph: &EGraph<AstNode<Op>, A>,
    state: (Id, Id),
  ) where
    <A as Analysis<AstNode<Op>>>::Data: ClassMatch,
  {
    if self.aus_by_state.contains_key(&state) {
      // we have already enumerated this state
      return;
    }

    self.aus_by_state.insert(state, BTreeSet::new());

    // println!("Enumerating over state {:?} in egraph", state,);
    let mut aus: BTreeSet<AU<Op, (Id, Id), Type>> = BTreeSet::new();

    let mut same = false;
    let mut different = false;

    let ops1 = egraph[state.0].nodes.iter().map(AstNode::as_parts);
    let ops2 = egraph[state.1].nodes.iter().map(AstNode::as_parts);

    for (op1, args1) in ops1 {
      if op1.is_liblearn_banned_op() {
        different = true;
        continue;
      }
      for (op2, args2) in ops2.clone() {
        if op2.is_liblearn_banned_op() {
          different = true;
          continue;
        }
        if op1 == op2 {
          same = true;
          if args1.is_empty() && args2.is_empty() {
            // FIXME: 从此只插入Hole，不插入op
            let new_expr = AstNode::leaf(op1.clone()).into();
            // let new_expr = PartialExpr::Hole(state);
            aus.insert(AU::new_with_expr(
              new_expr,
              self.liblearn_config.cost.clone(),
            ));
          } else {
            // println!("Processing op1 {:?} and op2 {:?}", op1, op2);
            // recursively enumerate the inputs to this rule.
            let inputs: Vec<_> =
              args1.iter().copied().zip(args2.iter().copied()).collect();

            for next_state in &inputs {
              self.enumerate_over_egraph(egraph, *next_state);
            }

            let smax_arity = self.max_arity;
            // let au_range: Vec<Vec<PartialExpr<Op, (Id, Id)>>> =
            // au_range.collect::<Vec<_>>();
            let new_aus = inputs.iter().map(|input| {
              let aus =
                self.aus_by_state[input].iter().cloned().collect::<Vec<_>>();
              let past_len = aus.len();
              let mut filtered_aus = vec![];
              for au in aus {
                if !filter_get_arg_aus(&au.expr)
                  || !filter_fused_vec_aus(op1.is_vector_op(), &au.expr)
                  || !au.expr().leaves_are_all_holes()
                {
                  if past_len == 1 {
                    // 加入Hole
                    let new_expr = PartialExpr::Hole(state);
                    filtered_aus.push(AU::new_with_expr(
                      new_expr,
                      self.liblearn_config.cost.clone(),
                    ));
                  }
                  continue;
                }

                filtered_aus.push(au);
              }
              // if aus.len() < past_len {
              //   info!(
              //     "Filtered out {} AUs, now: {}",
              //     past_len - aus.len(),
              //     aus.len()
              //   );
              // }
              // println!("filtered_aus: {:?}", filtered_aus);
              filtered_aus
            });
            info!("get_range_aus");
            let au_range = new_aus.collect::<Vec<_>>();
            let scheduler = Scheduler::new(
              self.clock_period,
              self.area_estimator.clone(),
              self.delay_estimator.clone(),
              self.bb_query.clone(),
            );
            let au_merger = AUMerger::new(
              op1.clone(),
              scheduler,
              self.liblearn_config.au_merge_mod,
              self.liblearn_config.cost,
              self.liblearn_config.sample_num,
              10,
            );
            let new_aus = au_merger
              .merge(au_range)
              .into_iter()
              .filter(|au| {
                smax_arity.map_or(true, |max_arity| {
                  au.unique_holes().len() <= max_arity
                })
              })
              .filter(|au| {
                // 过滤掉过大的
                let size = au.size();
                let vars = au.unique_holes().len();
                size <= self.liblearn_config.max_lib_size
                  && vars <= max_inputs(&self.ci_encoding_config) as usize
              });

            // println!("aus_state.len() is {}", self.aus_by_state.len());
            // info!("deduplicating new_aus last: {} ",
            // new_aus.clone().count());

            // let new_aus_dedu = self
            //   .deduplicate_from_candidates(new_aus.clone().
            // collect::<Vec<_>>()); new_aus_dedu.len());
            let start_total = Instant::now(); // 总的执行时间//
                                              // 打印new_aus的大小 为每一个新的模式生成一个AU
            aus.extend(new_aus.map(|au| {
              AU::new_with_expr(au, self.liblearn_config.cost.clone())
            }));
            // aus.extend(new_aus_dedu);

            debug!(
              "Total processing time: {:?}, lenth of new_aus and aus is {}",
              start_total.elapsed(),
              aus.len()
            ); // 总的处理时间
          }
        } else {
          different = true;
        }
      }
    }

    if same && different {
      let new_expr = PartialExpr::Hole(state);
      aus.insert(AU::new_with_expr(
        new_expr,
        self.liblearn_config.cost.clone(),
      ));
    }
    self.filter_aus(aus, state, egraph);
  }

  fn filter_aus<A: Analysis<AstNode<Op>> + Clone>(
    &mut self,
    mut aus: BTreeSet<AU<Op, (Id, Id), Type>>,
    state: (Id, Id),
    egraph: &EGraph<AstNode<Op>, A>,
  ) where
    <A as Analysis<AstNode<Op>>>::Data: ClassMatch,
  {
    if aus.is_empty() {
      let new_expr = PartialExpr::Hole(state);
      aus.insert(AU::new_with_expr(
        new_expr,
        self.liblearn_config.cost.clone(),
      ));
    } else {
      // If the two e-classes cannot co-occur in the same program, do not
      // produce an AU for them! We filter out the anti-unifications which
      // are just concrete expressions with no variables, and then convert
      // the contained states to pattern variables. The conversion takes
      // alpha-equivalent anti-unifications to the same value, effectively
      // discarding redundant anti-unifications.

      let learn_constants = self.learn_constants;
      let learn_trivial = self.learn_trivial;
      let banned_ops = &self.banned_ops;

      let mut filtered_multi_opmask_count = 0usize;
      let mut filtered_multi_opmask_examples = Vec::new();
      let nontrivial_aus = aus
        .iter()
        .filter(|au| learn_constants || au.expr.has_holes())
        .cloned()
        .map(|au| (normalize(au.expr().clone()), au.delay().clone()))
        .filter_map(|((au, num_vars, var2id), delay)| {
          // Here we filter out rewrites that don't actually simplify
          // anything. We say that an AU rewrite simplifies an
          // expression if it replaces that expression with a function
          // call that is strictly smaller than the original
          // expression.
          //
          // The size of a function call `f e_1 ... e_n` is size(e1) +
          // ... + size(e_n) + n + 1, as there are n applications and
          // the function's identifier `f`.
          //
          // The size of an expression e containing n subexpressions
          // e_1, ..., e_n is k_1 * size(e_1) + ... + k_n * size(e_n)
          // + size(e[x_1/e_1, ..., x_n/e_n]) - (k_1 + ... + k_n),
          // where k_i is the number of times e_i appears in e and
          // x_1, ..., x_n are variables.
          //
          // Because size(e_i) can be arbitrarily large, if any
          // variable k_i is greater than 1, the difference in size
          // between the function call and the original expression can
          // also be arbitrarily large. Otherwise, if k_1 = ... = k_n
          // = 1, the rewrite can simplify an expression if and only
          // if size(e[x_1/e_1, ..., x_n/e_n]) > 2n + 1. This
          // corresponds to an anti-unification containing at least n
          // + 1 nodes.
          // if learn_trivial
          //   || (self.meta_au_config.enable_meta_au
          //     && self.meta_au_config.learn_trivial)
          //   || num_vars < au.num_holes()
          //   || au.num_nodes() > 1 + num_vars
          let mut flag = true;
          let mut ty_map = HashMap::new();
          let mut ty_map_detailed_io = HashMap::new();

          if learn_trivial
            || (self.meta_au_config.enable_meta_au
              && self.meta_au_config.learn_trivial)
            || (self.find_packs && self.find_pack_config.learn_trivial)
            || self.find_packs
            //FIXME: 目前num_vars = au.num_holes()也可以被选入
            || num_vars <= au.num_holes()
            || au.num_nodes() > 1 + num_vars
          {
            if au.size() == 1 {
              return None;
            }
            // FIXME: 现在是debug模式！！！
            // if true {
            // println!("learn_trivial: {}, num_vars < au.num_holes(): {},
            // au.num_nodes() > num_vars: {}", learn_trivial, num_vars <
            // au.num_holes(), au.num_nodes() > num_vars);
            // 从var2id中取出变量对应的eclassId，
            // 从Id对应的eclass中拿到data作为变量类型
            // 在进行过滤的过程中，需要严格控制变量的类型，
            // 如果位宽难以合并，那么就直接返回None
            for (k, v) in var2id.iter() {
              let id = v.clone();
              let ty0 = egraph[id.0].data.get_type()[0]
                .parse::<Type>()
                .unwrap_or_else(|_| {
                  panic!(
                    "parse type error: {}",
                    egraph[id.0].data.get_type()[0]
                  );
                });
              let ty1 = egraph[id.1].data.get_type()[0]
                .parse::<Type>()
                .unwrap_or_else(|_| {
                  panic!(
                    "parse type error: {}",
                    egraph[id.1].data.get_type()[0]
                  );
                });

              // 如果不能合并，就直接return None
              if !AstNode::can_merge_types(&ty0, &ty1) {
                flag = false;
                break;
              }

              let ty_neglecting_width =
                AstNode::merge_types_neglecting_width(&ty0, &ty1);

              let ty_merged = AstNode::merge_types(&ty0, &ty1);
              if !ty_merged.is_state_type() {
                ty_map_detailed_io.insert(k.to_string(), ty_merged.clone());
              }
              let tys = vec![ty_neglecting_width];
              ty_map.insert(k.clone(), tys);
            }
            // 最终加入关于output节点的类型
            let output_ty = match au.clone() {
              PartialExpr::Node(ast_node) => {
                let tys = ast_node.operation().get_result_type();
                let ty = tys
                  .get(0)
                  .map(|t| {
                    t.parse::<Type>().unwrap_or_else(|_| {
                      panic!("parse type error: {}", t);
                    })
                  })
                  .unwrap_or_else(|| Type::default());
                ty
              }
              PartialExpr::Hole(_) => Type::default(),
            };
            if !output_ty.is_state_type() {
              ty_map_detailed_io
                .insert("output".to_string(), output_ty.clone());
            }
          } else {
            flag = false;
          }
          if flag {
            let ast = Pattern::from(au.clone()).ast;
            let new_expr = ast
              .iter()
              .map(|node| match node {
                egg::ENodeOrVar::ENode(ast_node) => {
                  let new_node = (*ast_node).clone();
                  new_node
                }
                egg::ENodeOrVar::Var(_) => Op::var(0),
              })
              .collect::<Vec<AstNode<Op>>>();
            let rec_expr: RecExpr<AstNode<Op>> = new_expr.into();
            let scheduler = Scheduler::new(
              self.clock_period,
              self.area_estimator.clone(),
              self.delay_estimator.clone(),
              self.bb_query.clone(),
            );
            let (lat_cpu, lat_acc, area) = scheduler.asap_schedule(&rec_expr);
            let latency_gain = lat_cpu - lat_acc;
            return Some(AUWithType {
              au: AU::new(
                au.clone(),
                delay,
                latency_gain,
                area,
                self.liblearn_config.cost.clone(),
              ),
              ty_map,
              detailed_io: ty_map_detailed_io,
            });
          } else {
            None
          }
        })
        // 进行fill，如果au中含有超过lib节点，就不加入
        .filter(|au| {
          let expr = &au.au.expr;
          let recexpr: RecExpr<_> =
            Expr::try_from(expr.clone()).unwrap().into();
          for node in recexpr.iter() {
            if node.operation().is_lib() {
              return false;
            }
          }
          true
        })
        .filter(|au| match au.au.expr.clone() {
          PartialExpr::Node(ast_node) => !banned_ops
            .iter()
            .any(|op| ast_node.operation().discriminant_eq(op)),
          PartialExpr::Hole(_) => true,
        })
        .filter(|au| {
          // 主要目的是，将类似于and 1i var 这种无用的模式过滤掉
          let expr = au.au.expr.clone();
          let mut op = Op::default();
          let mut children_ops = vec![];
          match expr {
            PartialExpr::Node(ast_node) => {
              op = ast_node.operation().clone();
              children_ops = ast_node
                .args()
                .iter()
                .map(|arg| {
                  match arg {
                    PartialExpr::Node(child_node) => {
                      child_node.operation().clone()
                    }
                    PartialExpr::Hole(var) => {
                      // 如果是变量，就返回一个默认的op
                      Op::make_rule_var(var.to_string())
                    }
                  }
                })
                .collect::<Vec<_>>();
            }
            PartialExpr::Hole(_) => {
              return true; // 如果是Hole，就不需要过滤
            }
          }
          op.is_useful_expr(children_ops.as_slice())
        })
        .filter(|au| {
          if !self.meta_au_config.enable_meta_au {
            true
          } else {
            let profile = meta_au_mask_profile(&au.au.expr);
            if let Some(reason) =
              meta_au_mask_filter_reason(&profile, self.meta_au_config)
            {
              filtered_multi_opmask_count += 1;
              if filtered_multi_opmask_examples.len() < 3 {
                filtered_multi_opmask_examples.push(format!(
                  "{}:{}",
                  reason,
                  Pattern::from(au.au.expr.clone())
                ));
              }
              false
            } else {
              true
            }
          }
        });
      // FIXME: 先把每次的去重ban掉吧，就把max_lib的限制拿出来用用
      let nontrivial_aus = nontrivial_aus.filter(|au| {
        // 我希望限制的是RecExpr中的节点数目，而不是AU的节点数目
        let recexpr: RecExpr<_> =
          Expr::try_from(au.au.expr.clone()).unwrap().into();
        let size = recexpr.len();
        size <= self.liblearn_config.max_lib_size
      });
      // // 最后使用deduplicate_from_candidates去重(只有当au的数目大于sample_num的1/
      // 2时才去重)
      // let nontrivial_aus = if aus.len() > self.liblearn_config.sample_num / 2
      // {   self.deduplicate_from_candidates(nontrivial_aus.
      // collect::<Vec<_>>()) } else {
      //   nontrivial_aus.collect()
      // };

      // info!(
      //   "length of nontrivial_aus is {}",
      //   nontrivial_aus.clone().count()
      // );
      let nontrivial_aus = if self.meta_au_config.enable_meta_au {
        // 不需要完善类型
        nontrivial_aus.into_iter().collect::<Vec<_>>()
      } else {
        nontrivial_aus
          .into_iter()
          .map(|au| {
            let mut new_au = au.clone();
            // 完善类型
            new_au = au_type_complete(&new_au);
            new_au
          })
          .collect::<Vec<_>>()
      };
      if self.meta_au_config.enable_meta_au && filtered_multi_opmask_count > 0 {
        println!(
          "        • meta AU learn filter summary: filtered_multi_opmask={}",
          filtered_multi_opmask_count
        );
        if !filtered_multi_opmask_examples.is_empty() {
          println!(
            "          filtered multi-opmask examples: {}",
            filtered_multi_opmask_examples.join(" || ")
          );
        }
      }
      self.aus.extend(nontrivial_aus.clone());
    }
    // 需要对aus也做一下过滤，去掉无意义的
    // let aus: BTreeSet<AU<Op, (Id, Id), Type>> = aus
    //   .into_iter()
    //   .filter(|au| {
    //     // 主要目的是，将类似于and 1i var 这种无用的模式过滤掉
    //     let expr = au.expr.clone();
    //     let mut op = Op::default();
    //     let mut children_ops = vec![];
    //     match expr {
    //       PartialExpr::Node(ast_node) => {
    //         op = ast_node.operation().clone();
    //         children_ops = ast_node
    //           .args()
    //           .iter()
    //           .map(|arg| {
    //             match arg {
    //               PartialExpr::Node(child_node) => {
    //                 child_node.operation().clone()
    //               }
    //               PartialExpr::Hole(var) => {
    //                 // 如果是变量，就返回一个默认的op
    //                 // 对var中的两个Id进行排序
    //                 if var.0 < var.1 {
    //                   Op::make_rule_var(format!("var_{}_{}", var.0, var.1))
    //                 } else {
    //                   Op::make_rule_var(format!("var_{}_{}", var.1, var.0))
    //                 }
    //               }
    //             }
    //           })
    //           .collect::<Vec<_>>();
    //       }
    //       PartialExpr::Hole(_) => {
    //         return true; // 如果是Hole，就不需要过滤
    //       }
    //     }
    //     if op.is_useful_expr(children_ops.as_slice()) {
    //       true
    //     } else {
    //       // println!(
    //       //   "op {:?}, child_ops: {:?}, we filter it out",
    //       //   op, children_ops
    //       // );
    //       false
    //     }
    //   })
    //   .collect();
    // // 如果aus为空，就插入一个Hole
    // if aus.is_empty() {
    //   let new_expr = PartialExpr::Hole(state);
    //   self
    //     .aus_by_state
    //     .get_mut(&state)
    //     .unwrap()
    //     .insert(AU::new_with_expr(
    //       new_expr,
    //       self.liblearn_config.cost.clone(),
    //     ));
    //   return;
    // }
    *self.aus_by_state.get_mut(&state).unwrap() = aus;
  }
}

/// 本函数是为了将子节点是get_arg，并且arg是external的au过滤掉
pub fn filter_get_arg_aus<Op, Type>(pe: &PartialExpr<Op, Type>) -> bool
where
  Op: OperationInfo + Clone + Ord,
  Type: Clone + Ord,
{
  match pe {
    PartialExpr::Node(node) => {
      if node.operation().is_get_from_arg() {
        let arg_pe = node.args()[0].clone();
        match arg_pe {
          PartialExpr::Node(arg_node) => {
            // 一定是arg节点
            // 如果是external的arg，就过滤掉
            return !arg_node.operation().is_external_arg();
          }
          PartialExpr::Hole(_) => {}
        }
      }
      true
    }
    _ => true,
  }
}

pub fn filter_fused_vec_aus<Op, Type>(
  is_vec_op: bool,
  pe: &PartialExpr<Op, Type>,
) -> bool
where
  Op: OperationInfo + Clone + Ord,
  Type: Clone + Ord,
{
  // is_vec_op是上层操作符的类型，PartialExpr的类型需要和is_vec_op对齐，
  // 因为这是一个recursive的过程，所以只需要看PE的第一个节点就行
  match pe {
    PartialExpr::Node(node) => {
      let pe_is_vec = node.operation().is_vector_op();
      pe_is_vec == is_vec_op
    }
    _ => true,
  }
}

pub fn au_type_complete<Op, Type>(
  au_with_type: &AUWithType<Op, Type>,
) -> AUWithType<Op, Type>
where
  Op: Arity
    + Clone
    + Debug
    + Default
    + Ord
    + DiscriminantEq
    + Hash
    + Sync
    + Send
    + Display
    + 'static
    + Teachable
    + OperationInfo,
  Type: Debug
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
  TypeSet<Type>: ClassMatch,
  AstNode<Op>: TypeInfo<Type>,
{
  // 针对每个au, 使用OperationInfo中的get_rtypes方法来完善类型
  let new_au_with_type = au_with_type.clone();
  let type_map = au_with_type.ty_map.clone();
  let mut pe = au_with_type.au.expr.clone();
  fn visit<
    Op: OperationInfo + Clone + Ord + Arity + Debug,
    Type: FromStr + Clone + Display,
  >(
    pe: &mut PartialExpr<Op, Var>,
    type_map: &HashMap<Var, Vec<Type>>,
  ) where
    AstNode<Op>: TypeInfo<Type>,
  {
    match pe {
      PartialExpr::Node(node) => {
        let ty_str = node.operation().get_result_type();
        if ty_str.is_empty() {
          // 如果没有类型，那么就需要进行类型推断
          // 首先visit子节点
          for arg in node.args_mut() {
            visit(arg, type_map);
          }
          // 然后取出每个子节点的类型
          let mut types = vec![];
          for arg in node.args() {
            if let PartialExpr::Hole(var) = arg {
              // 如果是变量，就取出类型
              if let Some(ty) = type_map.get(var) {
                types.push(ty[0].clone());
              }
            } else {
              // 如果不是变量，就取出类型
              let ty =
                arg.clone().node().unwrap().operation().get_result_type()[0]
                  .clone()
                  .parse::<Type>()
                  .unwrap_or_else(|_| {
                    panic!("parse type error");
                  });
              types.push(ty);
            }
          }
          // 之后计算类型
          // 用一个伪Id作为args
          let args = node
            .args()
            .iter()
            .enumerate()
            .map(|(i, _)| Id::from(i as usize))
            .collect::<Vec<_>>();
          let normal_astnode = AstNode::new(node.operation().clone(), args);
          let ty = normal_astnode.get_rtype(&types).to_string();
          node.operation_mut().set_result_type(vec![ty]);
        }
      }
      PartialExpr::Hole(_) => {}
    }
  }
  visit(&mut pe, &type_map);
  new_au_with_type
}

/// Replaces the metavariables in an anti-unification with pattern variables.
/// Normalizing alpha-equivalent anti-unifications produces identical
/// anti-unifications. Returns a pair of the anti-unification and the number of
/// unique variables it contains.
#[must_use]
pub fn normalize<Op: OperationInfo + Clone + Ord, T: Eq + Clone + Ord>(
  au: PartialExpr<Op, T>,
) -> (PartialExpr<Op, Var>, usize, HashMap<Var, T>) {
  let mut metavars = Vec::new();
  let mut var2_id = HashMap::new();
  let to_var = |metavar: T| {
    let index = metavars
      .iter()
      .position(|other| other == &metavar)
      .unwrap_or_else(|| {
        metavars.push(metavar.clone());
        metavars.len() - 1
      });

    let var: Var = format!("?x{index}")
      .parse()
      .unwrap_or_else(|_| unreachable!());
    var2_id.insert(var.clone(), metavar);
    PartialExpr::Hole(var)
  };
  let normalized = au.fill(to_var);
  (normalized, metavars.len(), var2_id)
}

#[allow(dead_code)]
fn patternize<Op>(au: &PartialExpr<Op, (Id, Id)>) -> Pattern<AstNode<Op>>
where
  Op: Arity + Clone + Display + Ord + Send + Sync + 'static + OperationInfo,
  AstNode<Op>: Language,
{
  let au = au.clone();
  au.fill(|(s1, s2)| {
    let var = format!("?s_{s1}_{s2}")
      .parse()
      .unwrap_or_else(|_| unreachable!());
    PartialExpr::Hole(var)
  })
  .into()
}

/// Converts an anti-unification into a partial expression which defines a new
/// named function and applies it to the metavariables in the anti-unification.
/// The new function reifies the anti-unification, replacing metavariables by
/// lambda arguments.
///
/// For example, the anti-unification
///
/// ```text
/// (* ?x (+ ?y 1))
/// ```
///
/// would be converted to the partial expression
///
/// ```text
/// (lib l_i (lambda (lambda (* $0 (+ $1 1))))
///  (apply (apply l_i ?y) ?x))
/// ```
///
/// assuming `name` is "foo".
#[must_use]
pub fn reify<Op, T, LA, LD>(
  ix: LibId,
  au: PartialExpr<Op, T>,
  clock_period: usize,
  area_estimator: LA,
  delay_estimator: LD,
  bb_query: BBQuery,
) -> PartialExpr<Op, T>
where
  Op: Clone
    + Default
    + Arity
    + Debug
    + Display
    + Ord
    + Send
    + Sync
    + Teachable
    + 'static
    + Hash
    + OperationInfo,
  AstNode<Op>: Language + Schedulable<LA, LD>,
  T: Eq + Clone + Hash + Debug + Ord,
{
  let mut metavars = Vec::new();

  // Replace every metavariable in this antiunification with a de
  // Bruijn-indexed variable.
  // Metavariables might be located inside lambdas. To deal with this,
  // the de Brujin index that we return is equal to the index of the
  // metavar, added to however many lambdas wrap the metavar at that
  // point.
  let mut fun = au.clone().fill_with_binders(|metavar, num_binders| {
    let index = metavars
      .iter()
      .position(|other: &(T, usize)| other.0 == metavar)
      .unwrap_or_else(|| {
        metavars.push((metavar, num_binders));
        metavars.len() - 1
      });
    let index = index + num_binders;

    let mut res = PartialExpr::Hole(index);

    for i in (0..num_binders).rev() {
      res = Op::apply(res, Op::var(i).into()).into();
    }

    res
  });

  // foo (\. \. $0 $2 ?hole) => foo (\. \. $0 $2 ?$2)
  //                                          ^ binders = 2

  // All the function variables
  let offset = metavars.len();

  let mut max_locals = 0;

  fun =
    fun.map_leaves_with_binders(|node, binders| match node.as_binding_expr() {
      Some(BindingExpr::Var(index)) if index.0 >= binders => {
        max_locals = std::cmp::max(max_locals, index.0 - binders + 1);
        Op::var(index.0 + offset).into()
      }
      _ => node.into(),
    });

  // foo (\. \. $0 $2 ?$2) => foo (\. \. $0 $3 ?$2)

  let mut fun = fun.fill(|index| Op::var(index).into());

  // Wrap that in a lambda-abstraction, one for each variable we introduced.
  for _ in 0..(metavars.len() + max_locals) {
    fun = Op::lambda(fun).into();
  }
  // Now apply the new function to the metavariables in reverse order so they
  // match the correct de Bruijn indexed variable.
  let mut body = Op::lib_var(ix).into();
  while let Some((metavar, binders)) = metavars.pop() {
    let mut fn_arg = PartialExpr::Hole(metavar);
    for _i in 0..binders {
      fn_arg = Op::lambda(fn_arg).into();
    }
    body = Op::apply(body, fn_arg).into();
  }

  for index in 0..max_locals {
    body = Op::apply(body, Op::var(index).into()).into();
  }

  PartialExpr::Node(
    BindingExpr::Lib(ix, fun, body, OrderedFloat(0.0), OrderedFloat(0.0), 0)
      .into(),
  )
}
