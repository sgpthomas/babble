use bitvec::vec;
use rand::prelude::*;
use serde::{de, Deserialize};
use std::{
  collections::{HashMap, HashSet},
  fmt::{format, Debug, Display},
  hash::Hash,
  str::FromStr,
  sync::Arc,
  time::Duration,
};

/// 用于扩展生成的库，如(x * y) + 1 和 (m / n) + 1, 会生成(?a f ?b) + 1 f =
/// select(idx, (+, *))
use crate::{
  au_filter::TypeAnalysis,
  bb_query::{self, BBInfo, BBQuery},
  extract::beam_pareto::{ISAXAnalysis, TypeInfo},
  learn::{self, AUWithType, LearnedLibraryBuilder},
  perf_infer::{self, expr_perf_infer},
  rewrites::TypeMatch,
  runner::{AUMergeMod, EnumMode, LiblearnConfig, LiblearnCost, OperationInfo},
  schedule::{Schedulable, Scheduler},
  Arity, AstNode, BindingExpr, DiscriminantEq, Expr, ParetoConfig, PartialExpr,
  Printable, Teachable,
};
use egg::{
  rewrite, EGraph, ENodeOrVar, Id, Pattern, RecExpr, Rewrite, Runner, Searcher,
  Symbol, Var,
};
use lexpr::print;
use log::debug;
use nom::lib;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub struct MetaAUConfig {
  /// 是否使用Meta_AU, 不进行序列化，由PhaseConfig控制
  #[serde(skip)]
  pub enable_meta_au: bool,
  /// 是否进行eclass pair剪枝
  pub prune_eclass_pair: bool,
  /// 是否学习所有可能的AU(要不要过滤掉一些trivial的AU)
  pub learn_trivial: bool,
  /// 保留多少个含有mask的Meta_AU
  pub num_meta_au_mask: usize,
  /// 至多允许一个pack里面出现多少个operation
  pub max_operation: usize,
  /// 一个Meta AU中最多允许多少个op_mask节点
  #[serde(default = "default_max_opmask_per_meta_au")]
  pub max_opmask_per_meta_au: usize,
  /// 是否允许op_mask节点嵌套
  #[serde(default)]
  pub allow_nested_opmask: bool,
  /// op_mask节点数占AU大小的最大比例
  #[serde(default = "default_max_opmask_ratio")]
  pub max_opmask_ratio: f64,
  /// 是否允许最终生成的op_pack跨多个资源族
  #[serde(default)]
  pub allow_cross_family_op_pack: bool,
}

fn default_max_opmask_per_meta_au() -> usize {
  1
}

fn default_max_opmask_ratio() -> f64 {
  0.4
}

impl Default for MetaAUConfig {
  fn default() -> Self {
    MetaAUConfig {
      enable_meta_au: false,
      prune_eclass_pair: true,
      learn_trivial: true,
      num_meta_au_mask: 100,
      max_operation: 5,
      max_opmask_per_meta_au: default_max_opmask_per_meta_au(),
      allow_nested_opmask: false,
      max_opmask_ratio: default_max_opmask_ratio(),
      allow_cross_family_op_pack: false,
    }
  }
}

fn au2expr<Op, T>(
  pe: (PartialExpr<Op, Var>, TypeMatch<T>),
  idx: usize,
) -> Expr<Op>
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
  T: Clone + Display,
{
  let cond = pe.1;
  match pe.0 {
    PartialExpr::Node(astnode) => {
      let operation = astnode.operation().clone();
      let args = astnode.args();
      let mut new_args = Vec::with_capacity(args.len());
      for arg in args {
        let expr: Expr<Op> = au2expr((arg.clone(), cond.clone()), idx);
        new_args.push(Arc::new(expr));
      }
      let node = AstNode::new(operation, new_args);
      node.into()
    }
    PartialExpr::Hole(var) => {
      // 首先，我们将PE转化为Expr只是为了转化成Recexpr进行delay计算，
      // 所以Hole可以不需要在意，将其作为一个叶节点处理就好，
      // 目前直接使用rulevar表示
      let ty = cond.type_map.get(&var).unwrap_or_else(|| {
        panic!("Type for variable {:?} not found in condition", var)
      });
      // 将Vec<T>转化为String
      let ty_str = ty
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<String>>()
        .join(",");
      let s = format!("{}_{}", var.to_string(), ty_str).to_string();
      let node = AstNode::leaf(Op::make_rule_var(s));
      node.into()
    }
  }
}

/// 需要新加了mask节点，搜索meta_au的过程中需要自己手动写searcher的实现，
/// 目前仿照pattern的写法，只不过记录一下Opmask的特例
/// 实现一个searcher
/// 目前只允许使用一个Opmask节点，
struct MetaAUOpSearcher<Op> {
  pub searcher: Pattern<AstNode<Op>>,
  pub mask_results: HashSet<Op>,
  pub var_results: HashMap<Var, HashSet<Id>>,
}

impl<Op> MetaAUOpSearcher<Op>
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
    + OperationInfo
    + Default
    + 'static,
{
  fn new(searcher: Pattern<AstNode<Op>>) -> Self {
    MetaAUOpSearcher {
      searcher,
      mask_results: HashSet::new(),
      var_results: HashMap::new(),
    }
  }

  fn search<T: Debug + Default + Clone + Ord + Hash>(
    &mut self,
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  ) where
    AstNode<Op>: TypeInfo<T>,
  {
    // 使用searcher进行搜索
    let searcher = self.searcher.clone();
    let search_results = searcher.search(egraph);
    let ast = searcher.ast.clone();
    // 记录有多少个Opmask节点，初始化results
    let mut opmask_count = 0;
    for node in ast.iter() {
      match node {
        ENodeOrVar::ENode(enode) => {
          if enode.operation().is_opmask() {
            opmask_count += 1;
          }
        }
        ENodeOrVar::Var(_) => {}
      }
    }
    if opmask_count > 1 {
      // 目前只支持表达式里有一个Opmask节点
      return;
    }
    // 如果表达式太短，直接返回
    if ast.len() < 2 {
      return;
    }
    for eclass in egraph.classes() {
      let id = eclass.id;
      let enode = ast.last().unwrap().clone();
      match enode {
        ENodeOrVar::ENode(enode) => {
          self.found_matched_eclasses(enode, id, egraph);
        }
        ENodeOrVar::Var(_) => {
          // 变量节点不需要处理
        }
      }
    }
    // 除此之外还需要拿到Var节点的Id
    let vars = self.searcher.vars();
    for var in vars {
      let var_id = var.clone();
      let mut var_set = HashSet::new();
      for result in &search_results {
        let substs = result.substs.clone();
        for subst in substs {
          if subst.get(var.clone()).is_some() {
            let var_id = subst.get(var.clone()).unwrap();
            var_set.insert(*var_id);
          }
        }
      }
      self.var_results.insert(var_id, var_set);
    }
  }

  fn found_matched_eclasses<T: Debug + Default + Clone + Ord + Hash>(
    &mut self,
    enode: AstNode<Op>,
    id: Id,
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  ) -> bool
  where
    AstNode<Op>: TypeInfo<T>,
  {
    let mut matched = false;
    for node in egraph[id].nodes.iter() {
      if node.args().len() != enode.args().len() {
        // 如果args的个数不一样，就直接跳过
        continue;
      }
      if node.operation() == enode.operation() || enode.operation().is_opmask()
      {
        // 相同之后可以继续深度匹配
        // 首先拿出ast中的下一个enode
        let mut found = true;
        for i in 0..enode.args().len() {
          let ast_arg = enode.args()[i];
          match self.searcher.ast[ast_arg].clone() {
            ENodeOrVar::ENode(en) => {
              // 拿到node的子节点
              let egraph_arg = node.args()[i];
              // 递归调用
              if !self.found_matched_eclasses(en.clone(), egraph_arg, egraph) {
                found = false;
                break;
              }
            }
            ENodeOrVar::Var(_) => {
              // 变量节点不需要处理
            }
          }
        }
        if found {
          matched = true;
          // 如果当前是Opmask节点，就将其加入到mask_results中
          if enode.operation().is_opmask() {
            // 这里需要将Opmask节点的操作符加入到mask_results中
            let op = node.operation().clone();
            // op也需要是arithmetic operation
            if !op.is_arithmetic() {
              continue; // 如果不是算术操作符，就跳过
            }
            self.mask_results.insert(op);
          }
          break;
        }
      }
    }
    matched
  }
}

fn add_op_pack<Op>(
  initial_pe: PartialExpr<Op, Var>,
  pack_op: Op,
  select_idx: usize,
) -> PartialExpr<Op, Var>
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
  // 一个递归函数，找到opmask节点，将其替换成op_select[idx](oppack, args)即可
  match initial_pe {
    PartialExpr::Node(astnode) => {
      let mut operation = astnode.operation().clone();
      let args = astnode.args();
      let mut new_args = Vec::with_capacity(args.len());
      // 如果当前节点是opmask节点，就将其替换成op_select
      if operation.is_opmask() {
        let pack_pe: PartialExpr<Op, Var> =
          PartialExpr::Node(AstNode::leaf(pack_op.clone()));
        operation = Op::make_op_select(select_idx);
        new_args.push(pack_pe);
      }
      for arg in args {
        let expr: PartialExpr<Op, Var> =
          add_op_pack(arg.clone(), pack_op.clone(), select_idx);
        new_args.push(expr);
      }
      let node = AstNode::new(operation, new_args);
      node.into()
    }
    PartialExpr::Hole(_) => {
      // 直接返回就行
      initial_pe
    }
  }
}

fn fill_specific_op<Op>(
  initial_pe: PartialExpr<Op, Var>,
  op: Op,
) -> PartialExpr<Op, Var>
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
  match initial_pe {
    PartialExpr::Node(astnode) => {
      let mut operation = astnode.operation().clone();
      let args = astnode.args();
      let mut new_args = Vec::with_capacity(args.len());
      if operation.is_opmask() {
        // 如果当前节点是opmask节点，就将其替换成具体的操作符
        operation = op.clone();
      }
      for arg in args {
        let expr: PartialExpr<Op, Var> =
          fill_specific_op(arg.clone(), op.clone());
        new_args.push(expr);
      }
      let node = AstNode::new(operation, new_args);
      node.into()
    }
    PartialExpr::Hole(_) => {
      // 直接返回就行
      initial_pe
    }
  }
}

#[derive(Debug, Clone)]
pub struct ExpandMessage<Op, T>
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Send
    + Sync
    + OperationInfo
    + Debug
    + 'static,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + 'static,
  AstNode<Op>: TypeInfo<T>,
{
  // pub all_au_rewrites: Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>,
  // pub conditions: HashMap<usize, TypeMatch<T>>,
  /// 每个lib对应多个searcher
  pub searchers: HashMap<usize, Vec<PartialExpr<Op, Var>>>,
  /// 每个lib对应只对应一个applier
  pub appliers: HashMap<usize, Pattern<AstNode<Op>>>,
  /// 每个lib对应一组rewrite_condition
  pub rewrites_conditions: HashMap<
    usize,
    Vec<(Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>, TypeMatch<T>)>,
  >,
  /// Number of matches observed while learning each lib. The extractor uses
  /// this as a reuse signal in the cost model.
  pub lib_search_results: HashMap<usize, usize>,
  /// Lib ids generated from meta AU/op_pack expansion.
  pub meta_libs: HashSet<usize>,
  // pub libs: HashMap<usize, (PartialExpr<Op, Var>, Pattern<AstNode<Op>>)>,
  // pub normal_au_count: usize,
  // pub meta_au_rewrites:
  //   HashMap<usize, Vec<Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>>>,
}

impl<Op, T> Default for ExpandMessage<Op, T>
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Send
    + Sync
    + OperationInfo
    + Debug
    + 'static,
  T: Debug
    + Default
    + Clone
    + PartialEq
    + Ord
    + Hash
    + Send
    + Sync
    + Display
    + 'static,
  AstNode<Op>: TypeInfo<T>,
{
  fn default() -> Self {
    ExpandMessage {
      // all_au_rewrites: Vec::new(),
      searchers: HashMap::new(),
      appliers: HashMap::new(),
      rewrites_conditions: HashMap::new(),
      lib_search_results: HashMap::new(),
      meta_libs: HashSet::new(),
      // libs: HashMap::new(),
      // normal_au_count: 0,
      // meta_au_rewrites: HashMap::new(),
    }
  }
}

impl<Op, T> ExpandMessage<Op, T>
where
  Op: Display
    + Hash
    + Clone
    + Ord
    + Teachable
    + Arity
    + Send
    + Sync
    + OperationInfo
    + Debug
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
    + Display
    + 'static,
  AstNode<Op>: TypeInfo<T>,
{
  pub fn new() -> Self {
    ExpandMessage::default()
  }

  pub fn libs(&self) -> Vec<usize> {
    self.rewrites_conditions.keys().cloned().collect()
  }

  pub fn get_exprs(&self) -> Vec<RecExpr<AstNode<Op>>> {
    let mut exprs = Vec::new();
    for lib_id in self.rewrites_conditions.keys() {
      // 首先随便拿出一个condition(同一个lib， condition是一样的)
      let condition = self
        .rewrites_conditions
        .get(lib_id)
        .unwrap()
        .first()
        .unwrap()
        .1
        .clone();
      // 拿出seacher
      let searchers = self.searchers.get(lib_id).unwrap();
      for searcher in searchers {
        let searcher_ast = Pattern::from(searcher.clone()).ast.clone();
        let new_expr = searcher_ast
          .iter()
          .map(|node| match node {
            egg::ENodeOrVar::ENode(ast_node) => {
              let new_node = (*ast_node).clone();
              new_node
            }
            egg::ENodeOrVar::Var(var) => {
              // 搜索var对应的类型信息
              let tys =
                condition.type_map.get(var).cloned().unwrap_or_default();
              let ty = if tys.is_empty() {
                T::default()
              } else {
                tys[0].clone()
              };
              let new_name = format!("{}:{}", var, ty);
              // AstNode::leaf(RdgEgg {
              //   op: RdgEggOp::RulerVar(Symbol::from(new_name)),
              //   info: RdgEggInfo {
              //     result_tys: tys.clone(),
              //     ..Default::default()
              //   },
              // })
              let mut op = Op::make_rule_var(new_name);
              op.set_result_type(tys.iter().map(|t| t.to_string()).collect());
              AstNode::leaf(op)
            }
          })
          .collect::<Vec<AstNode<Op>>>();
        let mut rec_expr: RecExpr<AstNode<Op>> = new_expr.into();
        expr_perf_infer(&mut rec_expr);
        exprs.push(rec_expr);
      }
    }
    exprs
  }

  pub fn update_message(
    &mut self,
    lib_id: usize,
    rewrites_conditions: Vec<(
      Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>,
      TypeMatch<T>,
    )>,
    searchers: Vec<PartialExpr<Op, Var>>,
    applier: Pattern<AstNode<Op>>,
  ) {
    // 更新对应的lib_id的rewrite和condition
    self.rewrites_conditions.insert(lib_id, rewrites_conditions);
    // 更新对应的lib_id的searcher
    self.searchers.insert(lib_id, searchers);
    // 更新对应的lib_id的applier
    self.appliers.insert(lib_id, applier);
  }

  pub fn insert_from_messages(&mut self, lib_id: usize, other_messages: &Self) {
    if self.rewrites_conditions.contains_key(&lib_id) {
      // 如果已经存在这个lib_id，就不需要更新了
      return;
    }
    // 将other_messages中的内容添加到当前的messages中
    if let Some(rewrites) = other_messages.rewrites_conditions.get(&lib_id) {
      self
        .rewrites_conditions
        .entry(lib_id)
        .or_default()
        .extend(rewrites.clone());
    }
    if let Some(searchers) = other_messages.searchers.get(&lib_id) {
      self
        .searchers
        .entry(lib_id)
        .or_default()
        .extend(searchers.clone());
    }
    if let Some(applier) = other_messages.appliers.get(&lib_id) {
      self.appliers.insert(lib_id, applier.clone());
    }
    if let Some(search_len) = other_messages.lib_search_results.get(&lib_id) {
      self.lib_search_results.insert(lib_id, *search_len);
    }
    if other_messages.meta_libs.contains(&lib_id) {
      self.meta_libs.insert(lib_id);
    }
  }

  pub fn extend_from_messages(&mut self, other_messages: &Self) {
    // 将other_messages中的内容添加到当前的messages中
    for (lib_id, rewrites) in other_messages.rewrites_conditions.iter() {
      if self.rewrites_conditions.contains_key(lib_id) {
        // 如果已经存在这个lib_id，就不需要更新了
        continue;
      }
      self
        .rewrites_conditions
        .insert(lib_id.clone(), rewrites.clone());
      let searchers = other_messages.searchers[lib_id].clone();
      self.searchers.insert(lib_id.clone(), searchers);
      let applier = other_messages.appliers[lib_id].clone();
      self.appliers.insert(lib_id.clone(), applier);
      if let Some(search_len) = other_messages.lib_search_results.get(lib_id) {
        self.lib_search_results.insert(lib_id.clone(), *search_len);
      }
      if other_messages.meta_libs.contains(lib_id) {
        self.meta_libs.insert(lib_id.clone());
      }
    }
  }

  pub fn retain_with_ids(&mut self, ids: &HashSet<usize>) {
    // 保留lib_id在ids中的rewrite和condition
    self
      .rewrites_conditions
      .retain(|lib_id, _| ids.contains(lib_id));
    // 保留lib_id在ids中的searcher
    self.searchers.retain(|lib_id, _| ids.contains(lib_id));
    // 保留lib_id在ids中的applier
    self.appliers.retain(|lib_id, _| ids.contains(lib_id));
    self
      .lib_search_results
      .retain(|lib_id, _| ids.contains(lib_id));
    self.meta_libs.retain(|lib_id| ids.contains(lib_id));
  }

  pub fn delete_lib(&mut self, lib_id: usize) {
    // 删除lib_id对应的rewrite和condition
    self.rewrites_conditions.remove(&lib_id);
    // 删除lib_id对应的searcher
    self.searchers.remove(&lib_id);
    // 删除lib_id对应的applier
    self.appliers.remove(&lib_id);
    self.lib_search_results.remove(&lib_id);
    self.meta_libs.remove(&lib_id);
  }
}

// expand函数的输入是一个EGraph，之后在这个EGraph上进行库学习拿到AU，
// 然后使用AU变化成的rewrite进行搜索，将搜索到的子节点打包处理，
// 在EGraph中加入新的节点
pub fn expand<Op, T, LA, LD>(
  egraph: EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  root: Id,
  config: ParetoConfig<LA, LD>,
  bb_query: bb_query::BBQuery,
  max_lib_id: usize,
) -> ExpandMessage<Op, T>
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
    + OperationInfo
    + BBInfo,
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
  println!(
    "       • before expand, eclass size: {}, egraph size: {}",
    egraph.classes().len(),
    egraph.total_size()
  );
  // egraph.dot().to_png("target/expand_before.png").unwrap();
  // 使用learned_library进行库学习，因为和vetorize的目标一致，
  // 所以直接套用vectorize的learn部分
  let expansion_lib_config = LiblearnConfig::new(
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
  // 第一次是包搜索模式
  let learned_lib = LearnedLibraryBuilder::default()
    .find_packs()
    .with_find_pack_config(config.find_pack_config.clone())
    .learn_constants(config.learn_constants)
    .max_arity(config.max_arity)
    .with_last_lib_id(max_lib_id)
    .with_liblearn_config(expansion_lib_config)
    .with_clock_period(config.clock_period)
    .with_bb_query(bb_query.clone())
    .build(&egraph, vec![root]);
  println!("        • expand::learned {} libs", learned_lib.size());

  // for lib in learned_lib.libs() {
  //   println!("lib: {}", lib);
  // }

  let learned_aus: Vec<_> = learned_lib
    .messages()
    .iter()
    .map(|msg| {
      // 将每个AU转化成PartialExpr<Op, Var>
      let expr: PartialExpr<Op, Var> = msg.searcher_pe.clone();
      let condition = msg.condition.clone();
      (expr, condition)
    })
    .collect();
  let mut meta_egraph = EGraph::new(ISAXAnalysis::new(
    config.final_beams,
    config.inter_beams,
    config.lps,
    BBQuery::default(),
  ));

  // // 将au生成的Recexpr加入到meta_egraph中
  let learned_messages: Vec<_> = learned_lib.messages();
  let mut roots = Vec::new();
  for i in (0..learned_aus.len()).rev() {
    let au = learned_aus[i].clone();
    // 在转化的过程中，需要考虑每一个Var如何转化，
    // 目前只是转化成了一个RuleVar，   // 但是不同au中的RuleVar理应是不一样的
    let expr = au2expr(au, i);
    let recexpr = RecExpr::from(expr);
    let root = meta_egraph.add_expr(&recexpr);
    roots.push(root);
    meta_egraph.rebuild();
  }

  // 加入list节点作为根节点
  let new_root = if roots.len() == 1 {
    roots[0]
  } else {
    let mut bbs = HashSet::new();
    for root in roots.iter() {
      bbs.extend(meta_egraph[*root].data.bb.clone());
    }
    let bbs = bbs.into_iter().collect::<Vec<_>>();
    let mut list_op = AstNode::new(Op::list(), roots.iter().copied());
    list_op.operation_mut().set_bbs_info(bbs);
    meta_egraph.add(list_op)
  };

  perf_infer::perf_infer(&mut meta_egraph, &vec![new_root]);

  // meta_egraph.dot().to_png("target/expand.png").unwrap();
  println!(
    "       • For meta_egraph, eclass size: {}, egraph size: {}",
    meta_egraph.classes().len(),
    meta_egraph.total_size()
  );

  let meta_au_lib_config = LiblearnConfig::new(
    LiblearnCost::Size,
    AUMergeMod::Boundary,
    if config.op_pack_config.prune_eclass_pair {
      EnumMode::PruningGold
    } else {
      EnumMode::All
    },
    // 后面的配置直接使用config.liblearn中的配置
    config.liblearn_config.sample_num,
    config.liblearn_config.hamming_threshold,
    config.liblearn_config.jaccard_threshold,
    config.liblearn_config.max_libs,
    config.liblearn_config.min_lib_size,
    config.liblearn_config.max_lib_size,
  );

  // 计算一个新的max_lib_id
  let max_lib_id = max_lib_id + learned_lib.size();

  // 进行meta_au-search
  let learn_meta_lib = LearnedLibraryBuilder::default()
    .learn_constants(config.learn_constants)
    .max_arity(config.max_arity)
    .with_liblearn_config(meta_au_lib_config)
    .with_clock_period(config.clock_period)
    .with_meta_au_config(config.op_pack_config)
    .with_last_lib_id(max_lib_id)
    .with_area_estimator(config.area_estimator.clone())
    .with_delay_estimator(config.delay_estimator.clone())
    .with_bb_query(bb_query.clone())
    .build(&meta_egraph, vec![new_root]);

  let meta_messages: Vec<_> = learn_meta_lib.messages();

  // 进行去重
  // learn_meta_lib.deduplicate(&meta_egraph);

  println!(
    "        • expand::learned {} meta libs",
    learn_meta_lib.size()
  );
  for lib in learn_meta_lib.libs() {
    println!("meta lib: {}", lib);
  }

  let mut rewrites_conditions: HashMap<
    usize,
    Vec<(Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>>, TypeMatch<T>)>,
  > = HashMap::new();
  let mut searchers: HashMap<usize, Vec<PartialExpr<Op, Var>>> = HashMap::new();
  let mut appliers: HashMap<usize, Pattern<AstNode<Op>>> = HashMap::new();
  let mut lib_search_results = HashMap::new();
  let mut meta_libs = HashSet::new();
  let mut skipped_no_mask = 0;
  let mut skipped_too_many_ops = 0;
  let mut skipped_too_few_ops = 0;
  let mut skipped_nonbeneficial = 0;
  let mut trimmed_cross_family = 0;
  let mut generated_meta_libs = 0;
  let mut generated_meta_rewrites = 0;
  let mut concrete_search_cache: HashMap<
    (PartialExpr<Op, Var>, Op),
    (PartialExpr<Op, Var>, Pattern<AstNode<Op>>, usize),
  > = HashMap::new();
  let mut concrete_search_cache_hits = 0;
  let mut concrete_search_cache_misses = 0;

  for msg in learned_messages {
    let lib_id = msg.lib_id;
    rewrites_conditions
      .entry(lib_id)
      .or_default()
      .push((msg.rewrite.clone(), msg.condition.clone()));
    // 将searcher_pe和applier_pe加入到对应的lib_id中
    searchers
      .entry(lib_id)
      .or_default()
      .push(msg.searcher_pe.clone());
    let applier: Pattern<_> = msg.applier_pe.into();
    appliers.insert(lib_id, applier);
    lib_search_results.insert(lib_id, msg.search_len);
  }

  for msg in meta_messages {
    let mut searcher = MetaAUOpSearcher::new(msg.searcher_pe.clone().into());
    // 使用searcher进行搜索
    searcher.search(&egraph);
    // 记录搜索到的结果
    let mask_results: HashSet<Op> = searcher.mask_results.clone();
    // TODO: 目前只对mask_results进行处理，后续可能需要对var_results进行处理
    // 第一步，拿到OpPack具体需要Pack的操作符
    if mask_results.len() == 0 {
      skipped_no_mask += 1;
      println!("        • meta lib {} skipped: no mask results", msg.lib_id);
      continue; // 如果没有mask结果，就跳过
    }
    let mut family_ops: HashMap<&'static str, Vec<(Op, String)>> =
      HashMap::new();
    for op in mask_results.iter() {
      let op_name = op.to_string();
      family_ops
        .entry(Op::op_pack_area_family(&op_name))
        .or_default()
        .push((op.clone(), op_name));
    }
    for ops in family_ops.values_mut() {
      ops.sort_by(|a, b| a.1.cmp(&b.1));
    }

    let allow_cross_family =
      config.op_pack_config.allow_cross_family_op_pack || family_ops.len() <= 1;
    let candidate_families = if allow_cross_family {
      family_ops.keys().copied().collect::<HashSet<_>>()
    } else {
      family_ops
        .iter()
        .filter_map(|(family, ops)| (ops.len() >= 2).then_some(*family))
        .collect::<HashSet<_>>()
    };
    if candidate_families.is_empty() {
      skipped_too_few_ops += 1;
      let mut family_breakdown = family_ops
        .iter()
        .map(|(family, ops)| {
          let op_names = ops
            .iter()
            .map(|(_, op_name)| op_name.clone())
            .collect::<Vec<_>>();
          format!("{family}={}", op_names.join("|"))
        })
        .collect::<Vec<_>>();
      family_breakdown.sort();
      println!(
        "        • meta lib {} skipped: cross-family trim leaves no family with >=2 ops ({})",
        msg.lib_id,
        family_breakdown.join(", ")
      );
      continue;
    }
    let candidate_op_count = candidate_families
      .iter()
      .filter_map(|family| family_ops.get(family))
      .map(Vec::len)
      .sum::<usize>();
    if allow_cross_family
      && candidate_op_count > config.op_pack_config.max_operation
    {
      skipped_too_many_ops += 1;
      let mut family_breakdown = family_ops
        .iter()
        .map(|(family, ops)| {
          let op_names = ops
            .iter()
            .map(|(_, op_name)| format!("{op_name}:not_searched"))
            .collect::<Vec<_>>();
          format!("{family}={}", op_names.join("|"))
        })
        .collect::<Vec<_>>();
      family_breakdown.sort();
      println!(
        "        • meta lib {} skipped: {} mask ops > max_operation {} ({})",
        msg.lib_id,
        candidate_op_count,
        config.op_pack_config.max_operation,
        family_breakdown.join(", ")
      );
      continue;
    }

    let mut family_matches: HashMap<&'static str, Vec<(Op, String, usize)>> =
      HashMap::new();
    for family in candidate_families.iter().copied() {
      let Some(ops) = family_ops.get(family) else {
        continue;
      };
      for (op, op_name) in ops {
        let key = (msg.searcher_pe.clone(), op.clone());
        let search_len =
          if let Some((_, _, search_len)) = concrete_search_cache.get(&key) {
            concrete_search_cache_hits += 1;
            *search_len
          } else {
            concrete_search_cache_misses += 1;
            let concrete_pe =
              fill_specific_op(msg.searcher_pe.clone(), op.clone());
            let searcher: Pattern<_> = concrete_pe.clone().into();
            let search_len = searcher.search(&egraph).len();
            concrete_search_cache
              .insert(key, (concrete_pe, searcher, search_len));
            search_len
          };
        family_matches.entry(family).or_default().push((
          op.clone(),
          op_name.clone(),
          search_len,
        ));
      }
    }

    let search_counts = family_matches
      .values()
      .flat_map(|ops| {
        ops
          .iter()
          .map(|(_, op_name, search_len)| (op_name.clone(), *search_len))
      })
      .collect::<HashMap<_, _>>();
    let mut family_breakdown = family_matches
      .iter()
      .map(|(family, ops)| {
        let mut op_names = ops
          .iter()
          .map(|(_, op_name, search_len)| {
            format!("{op_name}:matches={search_len}")
          })
          .collect::<Vec<_>>();
        op_names.sort();
        format!("{family}={}", op_names.join("|"))
      })
      .collect::<Vec<_>>();
    for (family, ops) in family_ops.iter() {
      if candidate_families.contains(family) {
        continue;
      }
      let mut op_names = ops
        .iter()
        .map(|(_, op_name)| {
          let search_len = search_counts.get(op_name);
          match search_len {
            Some(search_len) => format!("{op_name}:matches={search_len}"),
            None => format!("{op_name}:not_searched"),
          }
        })
        .collect::<Vec<_>>();
      op_names.sort();
      family_breakdown.push(format!("{family}={}", op_names.join("|")));
    }
    family_breakdown.sort();

    let mut selected_family = "cross_family".to_string();
    let selected_ops = if allow_cross_family {
      let mut ops = family_matches
        .into_iter()
        .flat_map(|(_, ops)| ops)
        .collect::<Vec<_>>();
      ops.sort_by(|a, b| a.1.cmp(&b.1));
      if family_breakdown.len() == 1 {
        selected_family = family_breakdown[0]
          .split('=')
          .next()
          .unwrap_or("")
          .to_string();
      }
      ops
    } else {
      trimmed_cross_family += 1;
      let mut families = family_matches.into_iter().collect::<Vec<_>>();
      families.sort_by(|(family_a, ops_a), (family_b, ops_b)| {
        let matches_a = ops_a
          .iter()
          .map(|(_, _, search_len)| search_len)
          .sum::<usize>();
        let matches_b = ops_b
          .iter()
          .map(|(_, _, search_len)| search_len)
          .sum::<usize>();
        matches_b
          .cmp(&matches_a)
          .then_with(|| ops_b.len().cmp(&ops_a.len()))
          .then_with(|| family_a.cmp(family_b))
      });
      let Some((family, mut ops)) = families
        .into_iter()
        .filter(|(_, ops)| ops.len() >= 2)
        .next()
      else {
        skipped_too_few_ops += 1;
        println!(
          "        • meta lib {} skipped: cross-family trim leaves no family with >=2 ops ({})",
          msg.lib_id,
          family_breakdown.join(", ")
        );
        continue;
      };
      selected_family = family.to_string();
      ops.sort_by(|a, b| a.1.cmp(&b.1));
      println!(
        "        • meta lib {} trimmed cross-family pack to family {} from {}",
        msg.lib_id,
        selected_family,
        family_breakdown.join(", ")
      );
      ops
    };

    if selected_ops.len() < 2 {
      skipped_too_few_ops += 1;
      println!(
        "        • meta lib {} skipped: {} mask op after family selection",
        msg.lib_id,
        selected_ops.len()
      );
      continue;
    }
    if selected_ops.len() > config.op_pack_config.max_operation {
      skipped_too_many_ops += 1;
      println!(
        "        • meta lib {} skipped: {} mask ops > max_operation {}",
        msg.lib_id,
        selected_ops.len(),
        config.op_pack_config.max_operation
      );
      continue; // 如果操作符的个数超过了限制，就直接跳过
    }
    // 打印lib
    // println!("Processing lib: {}", Pattern::from(msg.searcher_pe.clone()));
    // 否则新建一个OpPack节点和一个OpSelect节点
    let mut pack_bbs = Vec::new();
    for (op, _, _) in &selected_ops {
      pack_bbs.extend(op.get_bbs_info());
    }
    let mask_ops = selected_ops
      .iter()
      .map(|(_, op_name, _)| op_name.clone())
      .collect::<Vec<_>>();
    let op_pack = Op::make_op_pack(mask_ops.clone(), pack_bbs);
    // 将新的partial_expr转化成applier。这里的 idx 只用于给 lib body 一个
    // 代表性形状；每条 concrete rewrite 会在下面生成自己的 op_select[idx]。
    let mut default_applier: Pattern<_> =
      add_op_pack(msg.applier_pe.clone(), op_pack.clone(), 0).into();
    let new_searcher: Pattern<_> =
      add_op_pack(msg.searcher_pe.clone(), op_pack.clone(), 0).into();
    // println!("new_applier: {}", new_applier);
    // Calculate the gain and cost of the new applier
    let ast = &new_searcher.ast;
    let new_expr = ast
      .iter()
      .map(|node| match node {
        egg::ENodeOrVar::ENode(ast_node) => ast_node.clone(),
        egg::ENodeOrVar::Var(_) => Op::var(0),
      })
      .collect::<Vec<AstNode<Op>>>();
    let mut rec_expr: RecExpr<AstNode<Op>> = new_expr.into();
    let scheduler = Scheduler::new(
      config.clock_period,
      config.area_estimator.clone(),
      config.delay_estimator.clone(),
      bb_query.clone(),
    );
    expr_perf_infer(&mut rec_expr);
    let (lat_cpu, lat_acc, area) = scheduler.asap_schedule(&rec_expr);
    if area == 0 || lat_acc >= lat_cpu {
      skipped_nonbeneficial += 1;
      println!(
        "        • meta lib {} skipped: non-beneficial schedule lat_cpu={}, lat_acc={}, area={} ({})",
        msg.lib_id,
        lat_cpu,
        lat_acc,
        area,
        family_breakdown.join(", ")
      );
      continue;
    }
    for node in default_applier.ast.iter_mut() {
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
    let mut meta_search_len = 0;
    let mut concrete_rewrites = Vec::new();
    for (id, (op, _, search_len)) in selected_ops.iter().enumerate() {
      let key = (msg.searcher_pe.clone(), op.clone());
      let (concrete_pe, searcher, cached_search_len) =
        if let Some((concrete_pe, searcher, search_len)) =
          concrete_search_cache.get(&key)
        {
          concrete_search_cache_hits += 1;
          (concrete_pe.clone(), searcher.clone(), *search_len)
        } else {
          concrete_search_cache_misses += 1;
          let concrete_pe =
            fill_specific_op(msg.searcher_pe.clone(), op.clone());
          let searcher: Pattern<_> = concrete_pe.clone().into();
          let search_len = searcher.search(&egraph).len();
          concrete_search_cache
            .insert(key, (concrete_pe.clone(), searcher.clone(), search_len));
          (concrete_pe, searcher, search_len)
        };
      debug_assert_eq!(*search_len, cached_search_len);
      meta_search_len += search_len;
      concrete_rewrites.push(format!("{}:matches={}", op, search_len));
      let mut new_applier: Pattern<_> =
        add_op_pack(msg.applier_pe.clone(), op_pack.clone(), id).into();
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
      let rewrite: Rewrite<AstNode<Op>, ISAXAnalysis<Op, T>> = Rewrite::new(
        format!("{}_{}", msg.rewrite.name, id),
        searcher,
        new_applier.clone(),
      )
      .unwrap_or_else(|_| unreachable!());
      rewrites_conditions
        .entry(msg.lib_id.clone())
        .or_default()
        .push((rewrite, msg.condition.clone()));
      searchers
        .entry(msg.lib_id.clone())
        .or_default()
        .push(concrete_pe);
    }
    appliers.insert(msg.lib_id.clone(), default_applier.clone());
    // Meta AU expands one abstract op_mask into multiple concrete rewrites. The
    // aggregate match count is useful for diagnostics, but using it as the
    // lib's required reuse count over-penalizes meta libs in the extractor.
    lib_search_results.insert(msg.lib_id, 1);
    meta_libs.insert(msg.lib_id);
    generated_meta_libs += 1;
    generated_meta_rewrites += selected_ops.len();
    concrete_rewrites.sort();
    println!(
      "        • meta lib {} generated: mask_ops={}, rewrites={}, matches={}, lat_cpu={}, lat_acc={}, area={}",
      msg.lib_id,
      mask_ops.join("|"),
      selected_ops.len(),
      meta_search_len,
      lat_cpu,
      lat_acc,
      area
    );
    println!(
      "          pattern: {}",
      Pattern::<AstNode<Op>>::from(msg.searcher_pe.clone())
    );
    println!("          applier: {}", default_applier);
    println!(
      "          area_families: {}; selected_family={}",
      family_breakdown.join(", "),
      selected_family
    );
    println!(
      "          concrete_rewrites: {}",
      concrete_rewrites.join(", ")
    );
  }
  println!(
    "        • meta AU expand summary: generated_libs={}, generated_rewrites={}, skipped_no_mask={}, skipped_too_many_ops={}, skipped_too_few_ops={}, skipped_nonbeneficial={}, trimmed_cross_family={}, concrete_search_cache_hits={}, concrete_search_cache_misses={}",
    generated_meta_libs,
    generated_meta_rewrites,
    skipped_no_mask,
    skipped_too_many_ops,
    skipped_too_few_ops,
    skipped_nonbeneficial,
    trimmed_cross_family,
    concrete_search_cache_hits,
    concrete_search_cache_misses
  );
  ExpandMessage {
    rewrites_conditions,
    searchers,
    appliers,
    lib_search_results,
    meta_libs,
  }
  //   let var_results: HashMap<Var, HashSet<Id>> =
  // searcher.var_results.clone();   //
  // 接下来需要将mask_results中的节点进行打包处理   // 1. 合并普通变量结果和
  // mask 结果到一个 Vec 用 Option<Var> 表示：Some(var)   //
  // 是普通变量，None 表示 mask。   let mut entries: Vec<(Option<Var>,
  // HashSet<Id>)> = Vec::new();   let mut set_ids = HashSet::new();
  //   // 普通变量
  //   for (var, ids) in var_results {
  //     if ids.len() > config.op_pack_config.max_eclass_id {
  //       // 如果ids的个数超过了限制，就直接跳过
  //       continue;
  //     }
  //     let mut vec_ids = ids.clone().into_iter().collect::<Vec<_>>();
  //     vec_ids.sort_unstable();
  //     if set_ids.contains(&vec_ids) {
  //       continue; // 如果ids已经存在，就跳过
  //     }
  //     entries.push((Some(var.clone()), ids.clone()));
  //     set_ids.insert(vec_ids);
  //   }
  //   // mask 结果
  //   entries.push((None, mask_results.clone()));

  //   // 2. 对每个 entry 做 pack + select
  //   for (opt_var, ids) in entries {
  //     // 首先我要做一步检查，如果ids对应的节点全部都是args ==
  //     // 0的节点，那么完全没有必要pack 直接跳过
  //     let mut all_args_zero = true;
  //     for id in &ids {
  //       let nodes = &egraph[*id].nodes;
  //       for node in nodes {
  //         if node.args().len() > 0 {
  //           all_args_zero = false;
  //           break;
  //         }
  //       }
  //     }
  //     if all_args_zero {
  //       continue; // 如果ids对应的节点全部都是args ==
  // 0的节点，那么完全没有必要pack     }
  //     if ids.is_empty() {
  //       continue; // 如果某个 var 或 mask 没匹配到任何 id，就跳过
  //     }

  //     // 收集所有 operation 名称，去重排序
  //     let mut op_vec: Vec<String> = ids
  //       .iter()
  //       .flat_map(|&eid| egraph[eid].nodes.iter())
  //       .map(|n| n.operation().to_string())
  //       .collect();
  //     // 如果op_vec中操作符的个数超过5个，就直接跳过
  //     if op_vec.len() > config.op_pack_config.max_operation {
  //       continue;
  //     }
  //     op_vec.sort_unstable();
  //     op_vec.dedup();

  //     // 建立 op -> index 映射
  //     let op_idx: HashMap<_, _> = op_vec
  //       .iter()
  //       .enumerate()
  //       .map(|(i, op)| (op.clone(), i))
  //       .collect();

  //     // 插入 pack 节点
  //     let pack_args: Vec<Id> = ids.iter().cloned().collect();
  //     let pack_op = Op::make_op_pack(op_vec.clone());
  //     let pack_id = new_egraph.add(AstNode::new(pack_op, pack_args));

  //     // 插入 select 节点，并 union 回原 eclass
  //     for &eid in &ids {
  //       for node in &egraph[eid].nodes {
  //         let idx = *op_idx.get(&node.operation().to_string()).unwrap();
  //         let mut args = vec![pack_id];
  //         args.extend_from_slice(node.args());
  //         let select_node = AstNode::new(Op::make_op_select(idx), args);
  //         let new_id = new_egraph.add(select_node);
  //         new_egraph.union(eid, new_id);
  //       }
  //     }

  //     // 日志：打印当前处理的是哪个变量或 mask
  //     match &opt_var {
  //       Some(var) => debug!("处理变量 {:?} 的 pack/select", var),
  //       None => debug!("处理 MASK 的 pack/select"),
  //     }

  //     // 重建 egraph
  //     new_egraph.rebuild();
  //   }
  // }
  // // new_egraph.dot().to_png("target/expand_meta.png").unwrap();
}
