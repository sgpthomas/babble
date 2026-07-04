use crate::analysis::SimpleAnalysis;
use crate::extract::beam_pareto::ISAXAnalysis;
use crate::extract::beam_pareto::TypeInfo;
use crate::runner::OperationInfo;
use crate::teachable::BindingExpr;

use super::{super::teachable::Teachable, AstNode, Expr};
use crate::ast_node::Arity;
use crate::learn::Match;
use crate::learn::normalize;
use crate::schedule::Schedulable;
use egg::{EGraph, ENodeOrVar, Id, Language, Pattern, RecExpr, Searcher, Var};
use std::{
  collections::HashSet,
  convert::{TryFrom, TryInto},
  error::Error,
  fmt::{self, Debug, Display, Formatter},
  hash::Hash,
  sync::Arc,
};

/// A partial expression. This is a generalization of an abstract syntax tree
/// where subexpressions can be replaced by "holes", i.e., values of type `T`.
/// The type [`Expr<Op>`] is isomorphic to `PartialExpr<Op, !>`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash, Ord)]
pub enum PartialExpr<Op: OperationInfo + Clone + Ord, T: Clone + Ord> {
  /// A node in the abstract syntax tree.
  Node(AstNode<Op, Self>),
  /// A hole containing a value of type `T`.
  Hole(T),
}

impl<Op, T> PartialExpr<Op, T>
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
  AstNode<Op>: Language,
  T: Eq + Clone + Hash + Debug + Default + Ord,
{
  pub fn get_match<Type>(
    self,
    egraph: &EGraph<AstNode<Op>, ISAXAnalysis<Op, Type>>,
  ) -> Vec<Match>
  where
    Type: Debug + Default + Clone + Ord + Hash,
    AstNode<Op>: TypeInfo<Type>,
  {
    let pattern: Pattern<_> = normalize(self.clone()).0.into();
    // A key in `cache` is a set of matches
    // represented as a sorted vector.
    let mut key = vec![];

    for m in pattern.search(egraph) {
      for sub in m.substs {
        let actuals: Vec<_> = pattern.vars().iter().map(|v| sub[*v]).collect();
        let match_signature = Match::new(m.eclass, actuals);
        key.push(match_signature);
      }
    }

    key.sort();
    key
  }
  // 定义get_delay函数，输入是一个PartialExpr<Op, T>，输出是一个u32
  // Critical path delay (maybe use hls in the future)
  pub fn get_delay(&self) -> usize {
    // 首先将PE转化为Expr
    let expr: Expr<Op> = self.clone().try_into().unwrap();
    // 将Expr转化成RecExpr
    let rec_expr: RecExpr<AstNode<Op>> = expr.into();
    // 计算delay
    let mut node_delay: Vec<usize> = Vec::new();
    for node in &rec_expr {
      let op_delay = 1;
      let args_sum_delay = node
        .args()
        .iter()
        .map(|id| {
          let idx: usize = (*id).into();
          node_delay[idx]
        })
        .sum::<usize>();
      node_delay.push(op_delay + args_sum_delay);
    }
    *node_delay.last().unwrap_or(&0)
  }
}

impl<Op, T> PartialExpr<Op, T>
where
  Op: Teachable + OperationInfo + Clone + Ord,
  T: Clone + Ord,
{
  /// Same as [`Self::fill`], but also provides the number of outer binders
  /// to the function.
  pub fn fill_with_binders<U, F>(self, mut f: F) -> PartialExpr<Op, U>
  where
    F: FnMut(T, usize) -> PartialExpr<Op, U>,
    U: Clone + Ord,
  {
    self.fill_with_binders_helper(&mut f, 0)
  }

  fn fill_with_binders_helper<U, F>(
    self,
    f: &mut F,
    depth: usize,
  ) -> PartialExpr<Op, U>
  where
    F: FnMut(T, usize) -> PartialExpr<Op, U>,
    U: Clone + Ord,
  {
    match self {
      PartialExpr::Node(node) => {
        let binders = match node.as_binding_expr() {
          Some(BindingExpr::Lambda(_)) => depth + 1,
          _ => depth,
        };
        let node = node.map(|child| child.fill_with_binders_helper(f, binders));
        PartialExpr::Node(node)
      }
      PartialExpr::Hole(hole) => f(hole, depth),
    }
  }

  /// Replaces the leaves in the partial expression by applying a function to
  /// the operation of and number of binders above each leaf.
  #[must_use]
  pub fn map_leaves_with_binders<F>(self, mut f: F) -> Self
  where
    F: FnMut(AstNode<Op, Self>, usize) -> Self,
  {
    self.map_leaves_with_binders_mut(&mut f, 0)
  }

  fn map_leaves_with_binders_mut<F>(self, f: &mut F, depth: usize) -> Self
  where
    F: FnMut(AstNode<Op, Self>, usize) -> Self,
  {
    match self {
      PartialExpr::Node(node) => {
        if node.is_empty() {
          f(node, depth)
        } else {
          let binders = match node.as_binding_expr() {
            Some(BindingExpr::Lambda(_)) => depth + 1,
            _ => depth,
          };
          let node =
            node.map(|child| child.map_leaves_with_binders_mut(f, binders));
          PartialExpr::Node(node)
        }
      }
      hole @ PartialExpr::Hole(_) => hole,
    }
  }
}

impl<Op: OperationInfo + Clone + Ord, T: Eq + Hash + Clone + Ord>
  PartialExpr<Op, T>
{
  /// Returns the set of unique holes in the partial expression.
  #[must_use]
  pub fn unique_holes(&self) -> HashSet<&T> {
    let mut holes = HashSet::new();
    match self {
      PartialExpr::Node(node) => {
        for expr in node {
          holes.extend(expr.unique_holes());
        }
      }
      PartialExpr::Hole(hole) => {
        holes.insert(hole);
      }
    };
    holes
  }
}

impl<Op: OperationInfo + Clone + Ord, T: Clone + Ord> PartialExpr<Op, T> {
  /// Returns `true` if the partial expression is a node.
  #[must_use]
  pub fn is_node(&self) -> bool {
    matches!(self, Self::Node(_))
  }

  /// Total number of nodes in the partial expression.
  #[must_use]
  pub fn size(&self) -> usize {
    match self {
      PartialExpr::Node(node) => 1 + node.iter().map(Self::size).sum::<usize>(),
      PartialExpr::Hole(_) => 1,
    }
  }

  /// Returns the number of nodes in the partial expression, not including
  /// holes.
  #[must_use]
  pub fn num_nodes(&self) -> usize {
    match self {
      PartialExpr::Node(node) => {
        1 + node.iter().map(Self::num_nodes).sum::<usize>()
      }
      PartialExpr::Hole(_) => 0,
    }
  }

  /// Returns the number of holes in the partial expression.
  #[must_use]
  pub fn num_holes(&self) -> usize {
    match self {
      PartialExpr::Node(node) => {
        node.iter().map(Self::num_holes).sum::<usize>()
      }
      PartialExpr::Hole(_) => 1,
    }
  }

  /// Returns `true` if the partial expression is a hole.
  #[must_use]
  pub fn is_hole(&self) -> bool {
    matches!(self, Self::Hole(_))
  }

  /// Unwraps the [`Node`](Self::Node) `self` to produce the underlying
  /// [`AstNode`]. If `self` is a hole, produces [`None`].
  #[must_use]
  pub fn node(self) -> Option<AstNode<Op, Self>> {
    match self {
      PartialExpr::Node(node) => Some(node),
      PartialExpr::Hole(_) => None,
    }
  }

  /// Unwraps the [`Hole`](Self::Hole) `self` to produce the underlying value.
  /// if `self` is an AST node, produces [`None`].
  #[must_use]
  pub fn hole(self) -> Option<T> {
    match self {
      PartialExpr::Hole(hole) => Some(hole),
      PartialExpr::Node(_) => None,
    }
  }

  /// Returns `true` if `self` is a complete expression containing no holes.
  #[must_use]
  pub fn has_holes(&self) -> bool {
    match self {
      PartialExpr::Node(node) => node.iter().any(Self::has_holes),
      PartialExpr::Hole(_) => true,
    }
  }

  #[must_use]
  pub fn leaves_are_all_holes(&self) -> bool {
    match self {
      PartialExpr::Hole(_) => true, // 是一个 Hole，符合要求
      PartialExpr::Node(ast_node) => {
        // 如果是节点，递归检查所有子节点
        ast_node
          .args()
          .iter()
          .all(|child| child.leaves_are_all_holes())
      }
    }
  }

  #[must_use]
  pub fn has_vector_op(&self) -> bool {
    match self {
      PartialExpr::Node(node) => {
        if node.operation().is_vector_op() {
          return true;
        }
        node.iter().any(Self::has_vector_op)
      }
      PartialExpr::Hole(_) => false,
    }
  }

  /// Replaces the holes in a partial expression of type `PartialExpr<Op, T>`
  /// with partial expressions of type `PartialExpr<Op, U>` to produce a new
  /// partial expression of type `PartialExpr<Op, U>`. Each hole's replacement
  /// partial expression is determined by applying a function to its value.
  #[must_use]
  pub fn fill<U, F>(self, mut f: F) -> PartialExpr<Op, U>
  where
    F: FnMut(T) -> PartialExpr<Op, U>,
    U: Clone + Ord,
  {
    self.fill_mut(&mut f)
  }

  /// Helper for [`Self::fill`] which takes its closure by mutable reference.
  #[must_use]
  fn fill_mut<U, F>(self, f: &mut F) -> PartialExpr<Op, U>
  where
    F: FnMut(T) -> PartialExpr<Op, U>,
    U: Clone + Ord,
  {
    match self {
      PartialExpr::Node(node) => {
        let node = node.map(|child| child.fill_mut(f));
        PartialExpr::Node(node)
      }
      PartialExpr::Hole(hole) => f(hole),
    }
  }
}

impl<Op: OperationInfo + Clone + Ord> From<PartialExpr<Op, Var>>
  for Pattern<AstNode<Op>>
where
  AstNode<Op>: Language,
{
  fn from(partial_expr: PartialExpr<Op, Var>) -> Self {
    fn build<Op: OperationInfo + Clone + Ord>(
      pattern: &mut Vec<ENodeOrVar<AstNode<Op>>>,
      partial_expr: PartialExpr<Op, Var>,
    ) {
      match partial_expr {
        PartialExpr::Node(node) => {
          let (operation, args) = node.into_parts();
          let mut arg_ids = Vec::with_capacity(args.len());
          for arg in args {
            build(pattern, arg);
            arg_ids.push(Id::from(pattern.len() - 1));
          }
          pattern.push(ENodeOrVar::ENode(AstNode {
            operation,
            args: arg_ids,
          }));
        }
        PartialExpr::Hole(contents) => pattern.push(ENodeOrVar::Var(contents)),
      }
    }

    let mut pattern = Vec::new();
    build(&mut pattern, partial_expr);
    RecExpr::from(pattern).into()
  }
}

impl<Op: Clone + OperationInfo + Clone + Ord> From<Pattern<AstNode<Op>>>
  for PartialExpr<Op, Var>
{
  fn from(pattern: Pattern<AstNode<Op>>) -> Self {
    fn build<Op: Clone + OperationInfo + Ord>(
      pattern: &[ENodeOrVar<AstNode<Op>>],
    ) -> PartialExpr<Op, Var> {
      match &pattern[pattern.len() - 1] {
        &ENodeOrVar::Var(var) => PartialExpr::Hole(var),
        ENodeOrVar::ENode(node) => {
          let node = node.clone().map(|id| {
            let child_index = usize::from(id);
            build(&pattern[..=child_index])
          });
          PartialExpr::Node(node)
        }
      }
    }
    build(pattern.ast.as_ref())
  }
}

impl<Op: Clone + OperationInfo + Clone + Ord>
  From<RecExpr<ENodeOrVar<AstNode<Op>>>> for PartialExpr<Op, Var>
{
  fn from(expr: RecExpr<ENodeOrVar<AstNode<Op>>>) -> Self {
    fn build<Op: Clone + OperationInfo + Ord>(
      pattern: &[ENodeOrVar<AstNode<Op>>],
    ) -> PartialExpr<Op, Var> {
      match &pattern[pattern.len() - 1] {
        &ENodeOrVar::Var(var) => PartialExpr::Hole(var),
        ENodeOrVar::ENode(node) => {
          let node = node.clone().map(|id| {
            let child_index = usize::from(id);
            build(&pattern[..=child_index])
          });
          PartialExpr::Node(node)
        }
      }
    }
    build(expr.as_ref())
  }
}

/// An error which can be returned when attempting to convert a [`PartialExpr`]
/// to an [`Expr`], indicating that the partial expression is incomplete.
#[derive(Debug, Clone)]
pub struct IncompleteExprError<T> {
  /// The hole encountered.
  hole: T,
}

impl<T: Debug> Error for IncompleteExprError<T> {}
impl<T: Debug> Display for IncompleteExprError<T> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    write!(f, "expected expression but found hole {:?}", self.hole)
  }
}

// 修改从PE转化到Expr的函数，当PE是Hole的时候，不再返回错误，
// 而是返回一个叶子节点表示的表达式

impl<Op: Default + Arity + Debug + OperationInfo + Clone + Ord, T: Clone + Ord>
  TryFrom<PartialExpr<Op, T>> for Expr<Op>
{
  type Error = IncompleteExprError<T>;

  fn try_from(partial_expr: PartialExpr<Op, T>) -> Result<Self, Self::Error> {
    match partial_expr {
      PartialExpr::Node(AstNode { operation, args }) => {
        let mut new_args = Vec::with_capacity(args.len());
        for arg in args {
          let expr: Expr<Op> = arg.try_into()?;
          new_args.push(Arc::new(expr));
        }
        let node = AstNode {
          operation,
          args: new_args,
        };
        Ok(node.into())
      }
      PartialExpr::Hole(_) => {
        // 首先，我们将PE转化为Expr只是为了转化成Recexpr进行delay计算，
        // 所以Hole可以不需要在意，将其作为一个叶节点处理就好，
        // 目前直接使用rulevar表示
        let node = AstNode::leaf(Op::default());
        Ok(node.into())
      }
    }
  }
}

impl<Op: OperationInfo + Clone + Ord, T: Clone + Ord>
  TryFrom<PartialExpr<Op, T>> for AstNode<Op, PartialExpr<Op, T>>
{
  type Error = IncompleteExprError<T>;

  fn try_from(partial_expr: PartialExpr<Op, T>) -> Result<Self, Self::Error> {
    match partial_expr {
      PartialExpr::Node(node) => Ok(node),
      PartialExpr::Hole(hole) => Err(IncompleteExprError { hole }),
    }
  }
}

impl<Op: OperationInfo + Clone + Ord, T: Clone + Ord> From<AstNode<Op, Self>>
  for PartialExpr<Op, T>
{
  fn from(node: AstNode<Op, Self>) -> Self {
    Self::Node(node)
  }
}

impl<Op: Clone + OperationInfo + Ord, T: Clone + Ord> From<Expr<Op>>
  for PartialExpr<Op, T>
{
  fn from(expr: Expr<Op>) -> Self {
    Self::Node(AstNode::from(expr).map(|x| x.as_ref().clone().into()))
  }
}
