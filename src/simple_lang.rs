//! Defines [`SimpleOp`], a simple lambda calculus which can be used with
//! babble.

use std::{
  collections::{HashMap, HashSet},
  convert::Infallible,
  fmt::{self, Debug, Display, Formatter},
  str::FromStr,
  sync::Arc,
};

use egg::Symbol;
use strum::Display;

use crate::{
  DiscriminantEq, Expr, Precedence, Printer,
  ast_node::{Arity, AstNode, Memoize, Printable},
  extract::beam_pareto::{ClassMatch, TypeInfo, TypeSet},
  learn::LibId,
  runner::OperationInfo,
  schedule::Schedulable,
  teachable::{BindingExpr, DeBruijnIndex, Teachable},
};

/// Simplest language to use with babble
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimpleOp {
  /// A function application
  Apply,
  /// A de Bruijn-indexed variable
  Var(DeBruijnIndex),
  /// A reference to a lib fn
  LibVar(LibId),
  /// An uninterpreted symbol
  Symbol(Symbol),
  /// An anonymous function
  Lambda,
  /// A library function binding
  Lib(LibId, usize, usize),
  /// A list of expressions
  List,
  /// A add operation
  Add,
  /// A i64 constant
  Const(i64),
}

impl Arity for SimpleOp {
  fn min_arity(&self) -> usize {
    match self {
      Self::Var(_)
      | Self::LibVar(_)
      | Self::Symbol(_)
      | Self::Const(_)
      | Self::List => 0,
      Self::Lambda => 1,
      Self::Apply | Self::Lib(_, _, _) | Self::Add => 2,
    }
  }

  fn max_arity(&self) -> Option<usize> {
    match self {
      Self::Var(_) | Self::LibVar(_) | Self::Symbol(_) | Self::Const(_) => {
        Some(0)
      }
      Self::Lambda => Some(1),
      Self::Lib(_, _, _) | Self::Add => Some(2),
      _ => None,
    }
  }
}

impl Display for SimpleOp {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    let s = match self {
      Self::Apply => "@",
      Self::Lambda => "λ",
      Self::Lib(libid, _, _) => {
        return write!(f, "lib {libid}");
      }
      Self::LibVar(libid) => {
        return write!(f, "l{libid}");
      }
      Self::Var(index) => {
        return write!(f, "${index}");
      }
      Self::Symbol(sym) => {
        return write!(f, "{sym}");
      }
      Self::List => "list",
      Self::Add => "+",
      Self::Const(i) => {
        return write!(f, "{i}");
      }
    };
    f.write_str(s)
  }
}

impl Printable for SimpleOp {
  fn precedence(&self) -> Precedence {
    // all operations need to be surrounded with parens
    100
  }

  fn print_naked<W: std::fmt::Write + Default + Clone + ToString>(
    expr: Arc<Expr<Self>>,
    printer: &mut Printer<W>,
    memoizer: &mut dyn Memoize<Key = *const Expr<Self>>,
  ) -> std::fmt::Result {
    match (&expr.0.operation(), expr.0.args()) {
      (
        SimpleOp::Var(_)
        | SimpleOp::LibVar(_)
        | SimpleOp::Lambda
        | SimpleOp::Lib(_, _, _)
        | SimpleOp::Apply,
        _,
      ) => {
        write!(printer.writer, "<ERROR! binding-expr>")
      }
      (SimpleOp::List, ts) => printer.in_brackets(|p| {
        p.indented(&mut |p: &mut Printer<W>| {
          p.vsep(
            &mut |p: &mut Printer<W>, i: usize| {
              p.print_in_context(ts[i].clone(), 0, memoizer)
            },
            ts.len(),
            ",",
          )
        })
      }),
      (SimpleOp::Const(constant), []) => {
        write!(printer.writer, "{constant}")?;
        Ok(())
      }
      (SimpleOp::Symbol(symbol), []) => {
        write!(printer.writer, "{symbol}")?;
        Ok(())
      }
      (SimpleOp::Add, [a, b]) => {
        let args = [a, b];
        printer.chain(
          &mut |p: &mut Printer<W>| write!(p.writer, "add"),
          &mut |p: &mut Printer<W>| {
            p.indented(&mut |p: &mut Printer<W>| {
              p.vsep(
                &mut |p: &mut Printer<W>, i: usize| {
                  p.print(args[i].clone(), memoizer)
                },
                2,
                " ",
              )
            })
          },
          " ",
        )
      }
      (op, args) => {
        write!(printer.writer, "<ERROR! {op} with {} args>", args.len())
      }
    }?;

    Ok(())
  }
}

impl FromStr for SimpleOp {
  type Err = Infallible;

  fn from_str(input: &str) -> Result<Self, Self::Err> {
    let op: SimpleOp = match input {
      "apply" | "@" => Self::Apply,
      "lambda" | "λ" => Self::Lambda,
      "list" => Self::List,
      input => input
        .parse()
        .map(Self::Var)
        .or_else(|_| input.parse().map(Self::LibVar))
        // Lib is emitted
        .unwrap_or_else(|_| Self::Symbol(input.into())),
    };
    Ok(op)
  }
}

impl Teachable for SimpleOp {
  fn from_binding_expr<T>(binding_expr: BindingExpr<T>) -> AstNode<Self, T> {
    match binding_expr {
      BindingExpr::Lambda(body) => AstNode::new(Self::Lambda, [body]),
      BindingExpr::Apply(fun, arg) => AstNode::new(Self::Apply, [fun, arg]),
      BindingExpr::Var(index) => AstNode::leaf(Self::Var(index)),
      BindingExpr::Lib(ix, bound_value, body, latency, area) => {
        AstNode::new(Self::Lib(ix, latency, area), [bound_value, body])
      }
      BindingExpr::LibVar(ix) => AstNode::new(Self::LibVar(ix), []),
    }
  }

  fn as_binding_expr<T>(node: &AstNode<Self, T>) -> Option<BindingExpr<&T>> {
    let binding_expr = match node.as_parts() {
      (Self::Lambda, [body]) => BindingExpr::Lambda(body),
      (Self::Apply, [fun, arg]) => BindingExpr::Apply(fun, arg),
      (Self::Var(index), []) => BindingExpr::Var(*index),
      (Self::Lib(ix, latency, area), [bound_value, body]) => {
        BindingExpr::Lib(*ix, bound_value, body, *latency, *area)
      }
      (Self::LibVar(ix), []) => BindingExpr::LibVar(*ix),
      _ => return None,
    };
    Some(binding_expr)
  }

  fn list() -> Self {
    Self::List
  }
}

impl Schedulable for SimpleOp {
  fn op_latency(&self) -> usize {
    match self {
      Self::Apply => 0,
      Self::Lambda => 0,
      Self::Lib(_, _, _) => 0,
      Self::LibVar(_) => 0,
      Self::Var(_) => 0,
      Self::Symbol(_) => 0,
      Self::List => 0,
      Self::Add => 0,
      Self::Const(_) => 0,
    }
  }

  fn op_latency_cpu(&self) -> usize {
    match self {
      Self::Apply => 0,
      Self::Lambda => 0,
      Self::Lib(_, _, _) => 0,
      Self::LibVar(_) => 0,
      Self::Var(_) => 0,
      Self::Symbol(_) => 0,
      Self::List => 0,
      Self::Add => 1,
      Self::Const(_) => 0,
    }
  }

  fn op_area(&self) -> usize {
    match self {
      Self::Apply => 0,
      Self::Lambda => 0,
      Self::Lib(_, _, _) => 0,
      Self::LibVar(_) => 0,
      Self::Var(_) => 0,
      Self::Symbol(_) => 0,
      Self::List => 0,
      Self::Add => 1,
      Self::Const(_) => 0,
    }
  }

  fn op_delay(&self) -> usize {
    match self {
      Self::Apply => 0,
      Self::Lambda => 0,
      Self::Lib(_, _, _) => 0,
      Self::LibVar(_) => 0,
      Self::Var(_) => 0,
      Self::Symbol(_) => 0,
      Self::List => 0,
      Self::Add => 6,
      Self::Const(_) => 0,
    }
  }
}

impl Default for SimpleOp {
  fn default() -> Self {
    Self::List
  }
}

impl DiscriminantEq for SimpleOp {
  fn discriminant_eq(&self, other: &Self) -> bool {
    match (self, other) {
      (Self::Apply, Self::Apply) => true,
      (Self::Lambda, Self::Lambda) => true,
      (Self::LibVar(a), Self::LibVar(b)) => a == b,
      (Self::Var(a), Self::Var(b)) => a == b,
      (Self::Symbol(a), Self::Symbol(b)) => a == b,
      (Self::List, Self::List) => true,
      (Self::Add, Self::Add) => true,
      (Self::Const(a), Self::Const(b)) => a == b,
      (Self::Lib(a_id, a_lat, a_area), Self::Lib(b_id, b_lat, b_area)) => {
        a_id == b_id && a_lat == b_lat && a_area == b_area
      }
      _ => false,
    }
  }
}

impl OperationInfo for SimpleOp {
  fn get_libid(&self) -> usize {
    match self {
      Self::Lib(libid, _, _) => (*libid).0,
      _ => 0,
    }
  }

  fn is_lib(&self) -> bool {
    match self {
      Self::Lib(_, _, _) => true,
      _ => false,
    }
  }

  fn make_vec(result_tys: Vec<String>) -> Self {
    Self::List
  }
  fn make_get(id: usize, result_tys: Vec<String>) -> Self {
    Self::LibVar(LibId(id))
  }

  fn get_const(&self) -> Option<(i64, u32)> {
    match self {
      Self::Const(c) => Some((*c, 0)),
      _ => None,
    }
  }

  fn make_const(const_value: (i64, u32)) -> Self {
    Self::Const(const_value.0)
  }

  fn is_dummy(&self) -> bool {
    false
  }

  fn is_vector_op(&self) -> bool {
    false
  }

  fn get_simple_cost(&self) -> usize {
    0
  }

  fn make_gather(_: &Vec<usize>) -> Self {
    Self::List
  }

  fn make_shuffle(_: &Vec<usize>) -> Self {
    Self::List
  }
}

#[derive(Debug, Clone, Hash, PartialOrd, Ord, Display, Copy)]
pub enum SimpleType {
  #[strum(to_string = "int<{0}>")]
  IntT(usize),
  Unknown,
}

impl FromStr for SimpleType {
  type Err = Infallible;

  fn from_str(input: &str) -> Result<Self, Self::Err> {
    let op: SimpleType = match input {
      "int" => Self::IntT(0),
      _ => Self::Unknown,
    };
    Ok(op)
  }
}

impl Eq for SimpleType {}

impl PartialEq for SimpleType {
  fn eq(&self, other: &Self) -> bool {
    match (self, other) {
      (SimpleType::IntT(a), SimpleType::IntT(b)) => a == b,
      (SimpleType::Unknown, SimpleType::Unknown) => true,
      _ => false,
    }
  }
}

impl Default for SimpleType {
  fn default() -> Self {
    SimpleType::Unknown
  }
}

impl TypeInfo<SimpleType> for AstNode<SimpleOp> {
  fn get_rtype(&self, child_types: &Vec<SimpleType>) -> SimpleType {
    SimpleType::Unknown
  }

  fn merge_types(a: &SimpleType, b: &SimpleType) -> SimpleType {
    let a = match a {
      SimpleType::IntT(a) => *a,
      SimpleType::Unknown => 0,
    };
    let b = match b {
      SimpleType::IntT(b) => *b,
      SimpleType::Unknown => 0,
    };
    if a == 0 && b == 0 {
      SimpleType::Unknown
    } else if a == 0 {
      SimpleType::IntT(b)
    } else if b == 0 {
      SimpleType::IntT(a)
    } else {
      SimpleType::IntT(a.max(b))
    }
  }
}
