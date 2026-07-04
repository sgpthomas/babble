use super::{Arity, AstNode, ParseNodeError};
use crate::{runner::OperationInfo, sexp::Sexp, teachable::Teachable};
use egg::{Id, Language, RecExpr};
use std::{
  collections::HashMap, convert::TryFrom, hash::Hash, str::FromStr, sync::Arc,
};

/// An abstract syntax tree with operations `Op`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Expr<Op: OperationInfo + Ord>(pub AstNode<Op, Arc<Self>>);

#[allow(clippy::len_without_is_empty)]
impl<Op: OperationInfo + Ord> Expr<Op> {
  /// Converts `self` into its underlying [`AstNode`].
  #[must_use]
  pub fn into_inner(self) -> AstNode<Op, Arc<Self>> {
    self.0
  }

  /// Returns the number of AST nodes in the expression. There is no
  /// corresponding `is_empty` method because `len` is always greater than
  /// zero.
  #[must_use]
  pub fn len(&self) -> usize {
    self.0.iter().map(|x| x.len()).sum::<usize>() + 1
  }
}

impl<'a, Op: FromStr + Arity + OperationInfo + Ord> TryFrom<Sexp<'a>>
  for Expr<Op>
{
  type Error = ParseNodeError<Op, Arc<Expr<Op>>, <Op as FromStr>::Err>;

  fn try_from(sexp: Sexp<'a>) -> Result<Self, Self::Error> {
    let (op, args) = match sexp {
      Sexp::Atom(atom) => (atom, Vec::new()),
      Sexp::List(op, args) => (op, args),
    };
    let op: Op = op.parse().map_err(ParseNodeError::ParseError)?;
    let args = args
      .into_iter()
      .map(Self::try_from)
      .collect::<Result<Vec<_>, _>>()?
      .into_iter()
      .map(|x| Arc::new(x))
      .collect::<Vec<_>>();
    let node =
      AstNode::try_new(op, args).map_err(ParseNodeError::ArityError)?;
    Ok(Self(node))
  }
}

impl<Op: Hash + Eq + Clone + OperationInfo + Ord> From<Expr<Op>>
  for RecExpr<AstNode<Op>>
{
  fn from(expr: Expr<Op>) -> Self {
    fn build<Op: Hash + Eq + Clone + OperationInfo + Ord>(
      table: &mut HashMap<*const Expr<Op>, Id>,
      rec_expr: &mut Vec<AstNode<Op>>,
      rc_expr: Arc<Expr<Op>>,
    ) -> Id {
      let key = rc_expr.as_ref() as *const Expr<Op>;
      if let Some(id) = table.get(&key) {
        return *id;
      }
      let expr = rc_expr.as_ref().clone();
      let (operation, args) = expr.0.into_parts();
      let mut arg_ids = Vec::with_capacity(args.len());
      for arg in args {
        let id = build(table, rec_expr, arg);
        arg_ids.push(id);
      }
      rec_expr.push(AstNode {
        operation,
        args: arg_ids,
      });
      let id = Id::from(rec_expr.len() - 1);
      table.insert(key, id);
      id
    }
    let mut table = HashMap::new();
    let mut rec_expr = Vec::new();
    let _ = build(&mut table, &mut rec_expr, Arc::new(expr));
    rec_expr.into()
  }
}

impl<Op: Clone + std::fmt::Debug + OperationInfo + Ord>
  From<RecExpr<AstNode<Op>>> for Expr<Op>
{
  fn from(rec_expr: RecExpr<AstNode<Op>>) -> Self {
    fn build<Op: Clone + std::fmt::Debug + OperationInfo + Ord>(
      table: &mut HashMap<usize, Arc<Expr<Op>>>,
      rec_expr_slice: &[AstNode<Op>],
      id: usize,
    ) -> Arc<Expr<Op>> {
      if let Some(rc_expr) = table.get(&id) {
        return rc_expr.clone();
      }
      let node = rec_expr_slice[id].clone();
      let node = node.map(|id| {
        let child_index = usize::from(id);
        build(table, &rec_expr_slice, child_index)
      });
      let rc_expr = Arc::new(Expr(node));
      table.insert(id, rc_expr.clone());
      rc_expr
    }
    let mut table = HashMap::new();
    let rc_expr = build(&mut table, rec_expr.as_ref(), rec_expr.len() - 1);
    rc_expr.as_ref().clone()
  }
}

/// Convert a list of exprs into a single recexpr, combining them using the list
/// node
#[must_use]
pub fn combine_exprs<Op>(exprs: Vec<Expr<Op>>) -> RecExpr<AstNode<Op>>
where
  Op: Teachable
    + std::fmt::Debug
    + Clone
    + Arity
    + std::hash::Hash
    + Ord
    + OperationInfo,
{
  let mut res: Vec<AstNode<Op>> = Vec::new();
  let mut roots: Vec<egg::Id> = Vec::new();

  for expr in exprs {
    // Turn the expr into a RecExpr
    let recx: RecExpr<_> = expr.into();

    // Then turn the RecExpr into a Vec
    let mut nodes: Vec<AstNode<Op>> = recx.as_ref().to_vec();

    // For each node, increment the children by the current size of the accum
    // expr
    for node in &mut nodes {
      node.update_children(|x| (usize::from(x) + res.len()).into());
    }

    // Then push everything into the accum expr
    res.extend(nodes);
    roots.push((res.len() - 1).into());
  }

  // Add the root node
  res.push(AstNode::new(Op::list(), roots));

  // Turn res back into a recexpr!
  res.into()
}

impl<Op: OperationInfo + Ord> From<AstNode<Op, Arc<Self>>> for Expr<Op> {
  fn from(node: AstNode<Op, Arc<Self>>) -> Self {
    Self(node)
  }
}

impl<Op: OperationInfo + Ord> From<Expr<Op>> for AstNode<Op, Arc<Expr<Op>>> {
  fn from(expr: Expr<Op>) -> Self {
    expr.0
  }
}

impl<Op: OperationInfo + Ord> AsRef<AstNode<Op, Arc<Self>>> for Expr<Op> {
  fn as_ref(&self) -> &AstNode<Op, Arc<Self>> {
    &self.0
  }
}
