//! Abstract syntax trees.
use crate::runner::OperationInfo;
use egg::{FromOp, Id, Language};
use serde::{Deserialize, Serialize};
use std::{
  collections::HashSet,
  error::Error,
  fmt::{self, Debug, Display, Formatter},
  hash::Hash,
  slice,
  str::FromStr,
  vec,
};
use thiserror::Error;

/// An abstract syntax tree node representing an operation of type `Op` applied
/// to arguments of type `T`.
///
/// This type implements [`Language`] for arguments of type [`Id`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstNode<Op, T = Id> {
  operation: Op,
  args: Vec<T>,
}

impl<Op: PartialEq + OperationInfo, T: PartialEq + Clone + Ord> PartialEq
  for AstNode<Op, T>
{
  fn eq(&self, other: &Self) -> bool {
    // if the operation is list or vec , the args can be equai when they have
    // different order but the same element
    // if self.operation.is_vec() && other.operation.is_vec() {
    //   let mut self_args = self.args.clone();
    //   let mut other_args = other.args.clone();
    //   self_args.sort();
    //   other_args.sort();
    //   return self.operation == other.operation && self_args == other_args;
    // } else {
    return self.operation == other.operation && self.args == other.args;
    // }
  }
}

impl<Op: Eq + OperationInfo, T: Eq + Clone + Ord> Eq for AstNode<Op, T> {}

// 实现默认的PartialOrd和Ord
impl<Op: OperationInfo, T> PartialOrd for AstNode<Op, T>
where
  Op: PartialOrd,
  T: PartialOrd + Ord + Clone,
{
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    match self.operation.partial_cmp(&other.operation) {
      Some(std::cmp::Ordering::Equal) => self.args.partial_cmp(&other.args),
      ord => ord,
    }
  }
}

impl<Op: OperationInfo, T> Ord for AstNode<Op, T>
where
  Op: Ord,
  T: Ord + Clone,
{
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    match self.operation.cmp(&other.operation) {
      std::cmp::Ordering::Equal => self.args.cmp(&other.args),
      ord => ord,
    }
  }
}

impl<Op, T> Hash for AstNode<Op, T>
where
  Op: Hash + OperationInfo,
  T: Hash + Ord + Clone,
{
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    // if the operation is list or vec , the args can be equai when they have
    // different order but the same element
    self.operation.hash(state);
    // if self.operation.is_vec() {
    //   let mut args = self.args.clone();
    //   args.sort();
    //   args.hash(state);
    // } else {
    self.args.hash(state);
    // }
  }
}

pub use expr::{Expr, combine_exprs};
pub use partial_expr::PartialExpr;
pub use pretty::{Memoize, Precedence, Pretty, Printable, Printer};

mod expr;
mod partial_expr;
mod pretty;

/// A trait for operations which take a specific number of arguments.
pub trait Arity {
  /// Returns the minimum number of arguments the operation can take.
  fn min_arity(&self) -> usize;

  /// Returns the maximum number of arguments the operation can take, or
  /// [`None`] if there is no maximum.
  fn max_arity(&self) -> Option<usize> {
    Some(self.min_arity())
  }

  /// Returns `true` if the operation can take the given number of arguments.
  fn has_arity(&self, num_args: usize) -> bool {
    num_args >= self.min_arity()
      && self.max_arity().map_or(true, |max| num_args <= max)
  }
}

/// An error indicating that an operation was applied to the wrong number of
/// arguments.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArityError<Op, T> {
  /// The operation.
  operation: Op,
  /// The arguments.
  args: Vec<T>,
  /// The minimum allowed number of arguments.
  min: usize,
  /// The maximum allowed number of arguments.
  max: Option<usize>,
}

impl<Op: Debug, T: Debug> Error for ArityError<Op, T> {}

impl<Op: Debug, T> Display for ArityError<Op, T> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    write!(f, "the operation {:?} expects ", self.operation)?;
    match (self.min, self.max) {
      (min, Some(max)) if min == max => write!(f, "{max}")?,
      (0, Some(max)) => write!(f, "at most {max}")?,
      (min, Some(max)) => write!(f, "between {min} and {max}")?,
      (min, None) => write!(f, "at least {min}")?,
    };
    write!(f, " argument(s), but was given {}", self.args.len())
  }
}

impl<Op, T> AstNode<Op, T> {
  /// Returns the operation the node represents.
  #[must_use]
  pub fn operation(&self) -> &Op {
    &self.operation
  }

  #[must_use]
  pub fn operation_mut(&mut self) -> &mut Op {
    &mut self.operation
  }

  /// Returns a slice containing the operation's arguments.
  #[must_use]
  pub fn args(&self) -> &[T] {
    &self.args
  }

  /// Returns a slice which allows modifying the operation's arguments.
  #[must_use]
  pub fn args_mut(&mut self) -> &mut [T] {
    &mut self.args
  }

  /// Returns `true` if the operation has no arguments.
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.args.is_empty()
  }

  /// Returns the number of arguments the operation has.
  #[must_use]
  pub fn len(&self) -> usize {
    self.args.len()
  }

  /// Converts an `AstNode<Op, T>` into an `AstNode<Op, U>` by applying a
  /// function to each of its arguments.
  #[must_use]
  pub fn map<U, F>(self, f: F) -> AstNode<Op, U>
  where
    F: FnMut(T) -> U,
  {
    AstNode {
      operation: self.operation,
      args: self.args.into_iter().map(f).collect(),
    }
  }

  /// Returns an iterator over the operation's arguments.
  pub fn iter(&self) -> impl Iterator<Item = &T> {
    self.into_iter()
  }

  /// Returns an iterator that allows modifying the operation's arguments.
  pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
    self.into_iter()
  }

  /// Returns a reference to the node's operation and a slice of the operation's
  /// arguments.
  #[must_use]
  pub fn as_parts(&self) -> (&Op, &[T]) {
    (&self.operation, &self.args)
  }

  /// Decomposes the node into the operation and its arguments.
  #[must_use]
  pub fn into_parts(self) -> (Op, Vec<T>) {
    (self.operation, self.args)
  }
}

impl<Op: Arity + Debug, T> AstNode<Op, T> {
  /// Creates a node with the given operation and arguments.
  ///
  /// See also [`AstNode::into_parts`].
  ///
  /// # Panics
  ///
  /// Panics if the number of arguments does not match the
  /// [`Arity`] of the operation.
  #[must_use]
  pub fn new<I>(operation: Op, args: I) -> Self
  where
    I: IntoIterator<Item = T>,
  {
    match Self::try_new(operation, args) {
      Ok(node) => node,
      Err(e) => panic!("{}", e),
    }
  }

  /// Creates a leaf node with the given operation.
  ///
  /// # Panics
  ///
  /// Panics if the [`Arity`] of the operation cannot be zero.
  #[must_use]
  pub fn leaf(operation: Op) -> Self {
    Self::new(operation, [])
  }
}

impl<Op: Arity, T> AstNode<Op, T> {
  /// Creates a new node with the provided operation and arguments.
  ///
  /// # Errors
  ///
  /// Returns an error if the [`Arity`] of the operation doesn't match the
  /// number of arguments.
  pub fn try_new<I>(operation: Op, args: I) -> Result<Self, ArityError<Op, T>>
  where
    I: IntoIterator<Item = T>,
  {
    let args: Vec<_> = args.into_iter().collect();
    if operation.has_arity(args.len()) {
      Ok(Self { operation, args })
    } else {
      let (min, max) = (operation.min_arity(), operation.max_arity());
      Err(ArityError {
        operation,
        args,
        min,
        max,
      })
    }
  }
}

impl<Op, T> AsRef<[T]> for AstNode<Op, T> {
  /// Returns a reference to the operation's arguments.
  fn as_ref(&self) -> &[T] {
    self.args()
  }
}

impl<Op, T> AsRef<Op> for AstNode<Op, T> {
  /// Returns a reference to the node's operation.
  fn as_ref(&self) -> &Op {
    self.operation()
  }
}

impl<Op, T> AsMut<[T]> for AstNode<Op, T> {
  /// Returns a reference which allows modifying the operation's arguments.
  fn as_mut(&mut self) -> &mut [T] {
    self.args_mut()
  }
}

impl<'a, Op, T> IntoIterator for &'a AstNode<Op, T> {
  type Item = &'a T;

  type IntoIter = slice::Iter<'a, T>;

  fn into_iter(self) -> Self::IntoIter {
    self.args.iter()
  }
}

impl<'a, Op, T> IntoIterator for &'a mut AstNode<Op, T> {
  type Item = &'a mut T;

  type IntoIter = slice::IterMut<'a, T>;

  fn into_iter(self) -> Self::IntoIter {
    self.args.iter_mut()
  }
}

impl<Op, T> IntoIterator for AstNode<Op, T> {
  type Item = T;

  type IntoIter = vec::IntoIter<T>;

  /// Converts the node into an iterator over its arguments.
  fn into_iter(self) -> Self::IntoIter {
    self.args.into_iter()
  }
}

impl<Op> Language for AstNode<Op>
where
  Op: Ord + Debug + Clone + Hash + OperationInfo,
{
  type Discriminant = Op;

  fn matches(&self, other: &Self) -> bool {
    self.operation == other.operation && self.len() == other.len()
  }

  fn children(&self) -> &[Id] {
    self.args()
  }

  // Default methods

  fn children_mut(&mut self) -> &mut [Id] {
    self.args_mut()
  }

  fn for_each<F: FnMut(Id)>(&self, f: F) {
    self.iter().copied().for_each(f);
  }

  fn for_each_mut<F: FnMut(&mut Id)>(&mut self, f: F) {
    self.iter_mut().for_each(f);
  }

  fn try_for_each<E, F>(&self, f: F) -> Result<(), E>
  where
    F: FnMut(Id) -> Result<(), E>,
  {
    self.iter().copied().try_for_each(f)
  }

  fn len(&self) -> usize {
    self.len()
  }

  fn is_leaf(&self) -> bool {
    self.is_empty()
  }

  fn fold<F, T>(&self, init: T, f: F) -> T
  where
    F: FnMut(T, Id) -> T,
  {
    self.iter().copied().fold(init, f)
  }

  fn all<F>(&self, f: F) -> bool
  where
    F: FnMut(Id) -> bool,
  {
    self.iter().copied().all(f)
  }

  fn any<F>(&self, f: F) -> bool
  where
    F: FnMut(Id) -> bool,
  {
    self.iter().copied().any(f)
  }

  fn discriminant(&self) -> Self::Discriminant {
    self.operation().clone()
  }
}

/// An error which can be returned when parsing an expression using [`FromOp`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Error)]
pub enum ParseNodeError<Op, T, E> {
  /// The operator failed to parse.
  #[error(transparent)]
  ParseError(E),

  /// The operator was given the wrong number of arguments.
  #[error(transparent)]
  ArityError(ArityError<Op, T>),
}

impl<Op> FromOp for AstNode<Op>
where
  Op: Debug + Arity + FromStr + Clone + Ord + Hash + 'static + OperationInfo,
  <Op as FromStr>::Err: Error,
{
  type Error = ParseNodeError<Op, Id, <Op as FromStr>::Err>;

  fn from_op(operation: &str, args: Vec<Id>) -> Result<Self, Self::Error> {
    let operation = operation.parse().map_err(ParseNodeError::ParseError)?;
    Self::try_new(operation, args).map_err(ParseNodeError::ArityError)
  }
}

/// [Egg][egg] expects the [`Display`] implementation of a [`Language`] to
/// display only a node's operation, not its children. This implementation is
/// unexpected, so we only implement [`Display`] for the concrete type
/// [`AstNode<Op, Id>`].
impl<Op: Display> Display for AstNode<Op> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    self.operation.fmt(f)
  }
}
