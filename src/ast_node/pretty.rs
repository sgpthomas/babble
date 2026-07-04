use std::{
  collections::HashMap,
  fmt::{self, Display, Write},
  hash::Hash,
  sync::Arc,
};

use itertools::Itertools;

use crate::{
  ast_node::Expr,
  runner::OperationInfo,
  teachable::{BindingExpr, Teachable},
};

const LINE_LENGTH: usize = 80;
/// A wrapper around [`&'a Expr<Op>`] whose [`Display`] impl pretty-prints the
/// expression.
#[derive(Debug, Clone)]
pub struct Pretty<Op: OperationInfo + Clone + Ord> {
  expr: Arc<Expr<Op>>,
}

impl<Op: OperationInfo + Clone + Ord> Pretty<Op> {
  pub fn new(expr: Arc<Expr<Op>>) -> Self {
    Self { expr }
  }
}

impl<Op: OperationInfo + Clone + Ord> Display for Pretty<Op>
where
  Op: Printable + Teachable + Display,
{
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut printer = Printer::new(String::new());
    printer.print_in_context(
      self.expr.clone(),
      110,
      &mut ExprMemoizer::new(),
    )?;
    write!(f, "{}", printer.writer)
  }
}

#[test]
fn test_string_write() {
  let mut s = String::new();
  write!(s, "tuple").unwrap();
  assert_eq!(s, "tuple");
  write!(s, "a").unwrap();
  assert_eq!(s, "tuplea");
}

pub trait Memoize {
  type Key: Hash + Eq;
  fn register(&mut self, key: Self::Key) -> usize;
  fn lookup(&self, key: Self::Key) -> Option<usize>;
  fn next_id(&self) -> usize;
  fn checkpoint(&mut self) -> usize;
  fn restore(&mut self, id: usize);
}

#[derive(Debug, Clone)]
struct ExprMemoizer<Op: OperationInfo + Clone + Ord> {
  memo: HashMap<*const Expr<Op>, usize>,
  next_id: usize,
  checkpoints: Vec<HashMap<*const Expr<Op>, usize>>,
}

impl<Op: OperationInfo + Clone + Ord> ExprMemoizer<Op> {
  fn new() -> Self {
    Self {
      memo: HashMap::new(),
      next_id: 0,
      checkpoints: vec![],
    }
  }
}
impl<Op: OperationInfo + Clone + Ord> Memoize for ExprMemoizer<Op> {
  type Key = *const Expr<Op>;
  fn register(&mut self, key: Self::Key) -> usize {
    assert!(self.memo.get(&key).is_none(), "key already registered");
    let id = self.next_id;
    self.memo.insert(key, id);
    self.next_id += 1;
    id
  }

  fn lookup(&self, key: Self::Key) -> Option<usize> {
    self.memo.get(&key).copied()
  }
  fn next_id(&self) -> usize {
    self.next_id
  }
  fn checkpoint(&mut self) -> usize {
    let id = self.checkpoints.len();
    self.checkpoints.push(self.memo.clone());
    id
  }
  fn restore(&mut self, id: usize) {
    while self.checkpoints.len() > id {
      self.memo = self.checkpoints.pop().unwrap();
    }
  }
}

/// Operator precedence
pub type Precedence = u8;

/// A language whose expressions can be pretty-printed.
/// This is used for printing language-specific operations,
/// whereas the printing of binding expressions is implemented inside Printer.
pub trait Printable
where
  Self: Sized + OperationInfo + Clone + Ord,
{
  /// Operator precedence:
  /// determines whether the expression with head `op` will be parenthesized
  fn precedence(&self) -> Precedence;

  /// Print `expr` into the printer's buffer without parentheses
  ///
  /// # Errors
  ///
  /// This function returns an error if the underlying writer fails, or if the
  /// expression is malformed
  fn print_naked<
    W: Write + Clone + Default + ToString + Clone + Default + ToString,
  >(
    expr: Arc<Expr<Self>>,
    printer: &mut Printer<W>,
    memoizer: &mut dyn Memoize<Key = *const Expr<Self>>,
  ) -> fmt::Result;
}

/// Internal state of the pretty-printer
#[derive(Debug, Clone)]
pub struct Printer<
  W: Write + Clone + Default + ToString + Clone + Default + ToString,
> {
  /// Buffer where result is accumulated
  pub writer: W,
  /// Precedence level of the context
  /// (determines whether the next printed expression should be parenthesized)
  pub ctx_precedence: Precedence,
  /// Bound variables in current scope
  bindings: Vec<String>,
  // Current indentation level
  pub indentation: usize,
  backup_writer: W,
}

impl<W: Write + Clone + Default + ToString + Clone + Default + ToString>
  Printer<W>
{
  /// Create a fresh printer for the top-level expression
  fn new(writer: W) -> Self {
    Self {
      writer,
      bindings: vec![],
      ctx_precedence: 0,
      indentation: 0,
      backup_writer: W::default(),
    }
  }

  /// Print `expr` into the buffer at the current precedence level
  ///
  /// # Errors
  ///
  /// This function returns an error if the underlying writer fails, or if the
  /// expression is malformed
  pub fn print<Op: Printable + Teachable + Display>(
    &mut self,
    expr: Arc<Expr<Op>>,
    memoizer: &mut dyn Memoize<Key = *const Expr<Op>>,
  ) -> fmt::Result {
    let op = expr.0.operation();
    let old_prec = self.ctx_precedence;
    let new_prec = op.precedence();
    self.ctx_precedence = new_prec;
    if new_prec <= old_prec {
      self.writer.write_char('(')?;
      self.print_memoized(expr, memoizer)?;
      self.writer.write_char(')')?;
    } else {
      self.print_memoized(expr, memoizer)?;
    }
    self.ctx_precedence = old_prec;
    Ok(())
  }

  /// Print `expr` into the buffer at precedence level `prec`: this function
  /// is used to implement associativity and bracket-like expressions, where
  /// the children should be printed at a lower precedence level than the
  /// expression itself
  ///
  /// # Errors
  ///
  /// This function returns an error if the underlying writer fails, or if the
  /// expression is malformed
  pub fn print_in_context<Op: Printable + Teachable + Display>(
    &mut self,
    expr: Arc<Expr<Op>>,
    prec: Precedence,
    memoizer: &mut dyn Memoize<Key = *const Expr<Op>>,
  ) -> fmt::Result {
    let old_prec = self.ctx_precedence;
    self.ctx_precedence = prec;
    self.print(expr, memoizer)?;
    self.ctx_precedence = old_prec;
    Ok(())
  }

  fn print_memoized<Op: Printable + Teachable + Display>(
    &mut self,
    expr: Arc<Expr<Op>>,
    memoizer: &mut dyn Memoize<Key = *const Expr<Op>>,
  ) -> fmt::Result {
    if let Some(id) = memoizer.lookup(expr.as_ref() as *const _) {
      write!(self.writer, "[{id}]")
    } else {
      let id = memoizer.register(expr.as_ref() as *const _);
      self.print_naked(expr, memoizer, id)
    }
  }

  /// Print `expr` into the buffer (without parentheses)
  ///
  /// # Errors
  ///
  /// This function returns an error if the underlying writer fails, or if the
  /// expression is malformed.
  fn print_naked<Op: Printable + Teachable + Display>(
    &mut self,
    expr: Arc<Expr<Op>>,
    memoizer: &mut dyn Memoize<Key = *const Expr<Op>>,
    id: usize,
  ) -> fmt::Result {
    self.writer.write_str(&format!("[{id}]:"))?;

    match expr.0.as_binding_expr() {
      Some(binding_expr) => {
        match binding_expr {
          BindingExpr::Lambda(body) => {
            self.writer.write_char('λ')?;
            self.print_abstraction(body.clone(), memoizer)
          }
          BindingExpr::Apply(fun, arg) => {
            self.print_in_context(
              fun.clone(),
              self.ctx_precedence - 1,
              memoizer,
            )?; // app is left-associative
            self.writer.write_char(' ')?;
            self.print(arg.clone(), memoizer)
          }
          BindingExpr::Var(index) => {
            let name = self
              .bindings
              .get(self.bindings.len() - 1 - index.0)
              .expect("unbound variable");
            self.writer.write_str(name)
          }
          BindingExpr::Lib(ix, def, body, _, _, _) => {
            self.with_binding("f", |p| {
              write!(p.writer, "lib {ix} =")?; // print binding
              p.indentation += 1;
              p.new_line()?;
              p.print_in_context(def.clone(), 0, memoizer)?;
              p.indentation -= 1;
              p.new_line()?;
              p.writer.write_str("in")?;
              p.indentation += 1;
              p.new_line()?;
              p.print_in_context(body.clone(), 0, memoizer)?;
              p.indentation -= 1;
              Ok(())
            })
          }
          BindingExpr::LibVar(ix) => {
            write!(self.writer, "{ix}")
          }
        }
      }
      None => {
        // This is not a binding expr: use language-specific printing
        Op::print_naked(expr.clone(), self, memoizer)
      }
    }
  }

  /// Print abstraction with body `body` without the "λ" symbol
  /// (this implements the syntactic sugar with nested abstractions)
  ///
  /// # Errors
  ///
  /// This function returns an error if the underlying writer fails, or if the
  /// expression is malformed.
  fn print_abstraction<Op: Printable + Teachable + Display>(
    &mut self,
    body: Arc<Expr<Op>>,
    memoizer: &mut dyn Memoize<Key = *const Expr<Op>>,
  ) -> fmt::Result {
    self.with_binding("x", |p| {
      let fresh_var = p.bindings.last().unwrap(); // the name of the latest binding
      write!(p.writer, "{fresh_var} ")?; // print binding
      if let Some(BindingExpr::Lambda(inner_body)) = body.0.as_binding_expr() {
        p.print_abstraction(inner_body.clone(), memoizer) // syntactic sugar: no λ needed here
      } else {
        p.writer.write_str("-> ")?; // done with the sequence of bindings: print ->
        p.print_in_context(body.clone(), 0, memoizer) // body doesn't need parens
      }
    })
  }

  /// Chain printing
  pub fn chain<
    T1: FnMut(&mut Self) -> fmt::Result,
    T2: FnMut(&mut Self) -> fmt::Result,
  >(
    &mut self,
    f1: &mut T1,
    f2: &mut T2,
    sep: &str,
  ) -> fmt::Result {
    f1(self)?;
    write!(self.writer, "{sep}")?;
    f2(self)
  }

  /// Add new line with current indentation
  ///
  /// # Errors
  ///
  /// This function returns an error if the underlying writer fails.
  pub fn new_line(&mut self) -> fmt::Result {
    write!(self.writer, "\n{}", " ".repeat(self.indentation * 2))
  }

  /// Print f(i) for i in 0..n on separate lines
  ///
  /// # Errors
  ///
  /// This function returns an error if the underlying writer fails, or if the
  /// closure returns an error.
  pub fn vsep<T: FnMut(&mut Self, usize) -> fmt::Result>(
    &mut self,
    f: &mut T,
    n: usize,
    sep: &str,
  ) -> fmt::Result {
    for i in 0..n {
      f(self, i)?;
      if i < n - 1 {
        write!(self.writer, "{sep}")?;
        self.new_line()?;
      }
    }
    Ok(())
  }

  // swap writer and backup_writer, return the new backup_writer (cloned)
  fn swap_writer(&mut self) {
    std::mem::swap(&mut self.writer, &mut self.backup_writer);
  }
  /// backup writer to backup_writer, and write to an empty writer, then swap
  /// them at the ending
  pub fn temp<T: FnMut(&mut Self) -> fmt::Result>(
    &mut self,
    mut f: T,
  ) -> Result<W, fmt::Error> {
    let old_backup_writer = std::mem::take(&mut self.backup_writer);
    self.swap_writer();
    f(self)?;
    self.swap_writer();
    Ok(std::mem::replace(
      &mut self.backup_writer,
      old_backup_writer,
    ))
  }

  /// Print f(i) for i in 0..n on one/multiple lines
  /// if the original printing is too long or multi-lined, add newline
  /// adaptably.
  pub fn sep_adaptable<
    Op: OperationInfo + Ord + Clone,
    T: FnMut(
      &mut Self,
      usize,
      &mut dyn Memoize<Key = *const Expr<Op>>,
    ) -> fmt::Result,
  >(
    &mut self,
    f: &mut T,
    n: usize,
    sep: &str,
    memoizer: &mut dyn Memoize<Key = *const Expr<Op>>,
    adaptable: bool,
  ) -> fmt::Result {
    let backup_memoizer = memoizer.checkpoint();
    let before_temp_str = self.writer.to_string();

    let temp_printed_str = self
      .temp(|p| {
        for i in 0..n {
          f(p, i, memoizer)?;
          if i < n - 1 {
            write!(p.writer, "{sep}")?;
          }
        }
        Ok(())
      })?
      .to_string();

    let after_temp_string = self.writer.to_string();

    if before_temp_str != after_temp_string {
      log::error!(
        "temp changes writer!--------\n{}\n--------------------\n{}----------------------",
        before_temp_str,
        after_temp_string
      );
    }

    if adaptable
      && (temp_printed_str.lines().collect_vec().len() > 1
        || temp_printed_str.len() > LINE_LENGTH)
    {
      memoizer.restore(backup_memoizer);
      for i in 0..n {
        f(self, i, memoizer)?;
        if i < n - 1 {
          write!(self.writer, "{sep}")?;
          self.new_line()?;
        }
      }
    } else {
      write!(self.writer, "{temp_printed_str}")?;
    }
    Ok(())
  }

  /// print `f()` in parentheses
  ///
  /// # Errors
  ///
  /// This function returns an error if the underlying writer fails, or if the
  /// closure returns an error.
  pub fn in_parens<T: Fn(&mut Self) -> fmt::Result>(
    &mut self,
    f: T,
  ) -> fmt::Result {
    self.writer.write_char('(')?;
    f(self)?;
    self.writer.write_char(')')
  }

  /// print `f()` in brackets
  ///
  /// # Errors
  ///
  /// This function returns an error if the underlying writer fails, or if the
  /// closure returns an error.
  pub fn in_brackets<T: FnMut(&mut Self) -> fmt::Result>(
    &mut self,
    mut f: T,
  ) -> fmt::Result {
    self.writer.write_char('[')?;
    f(self)?;
    self.writer.write_char(']')
  }

  /// print `f()` indented one more level
  ///
  /// # Errors
  ///
  /// This function returns an error if the underlying writer fails, or if the
  /// closure returns an error.
  pub fn indented<T: FnMut(&mut Self) -> fmt::Result>(
    &mut self,
    f: &mut T,
  ) -> fmt::Result {
    self.indentation += 1;
    f(self)?;
    self.indentation -= 1;
    Ok(())
  }

  /// print `f()` inside the scope of a binder
  ///
  /// # Errors
  ///
  /// This function returns an error if the underlying writer fails, or if the
  /// closure returns an error.
  fn with_binding<T: FnMut(&mut Self) -> fmt::Result>(
    &mut self,
    prefix: &str,
    mut f: T,
  ) -> fmt::Result {
    self
      .bindings
      .push(format!("{prefix}{}", self.bindings.len()));
    f(self)?;
    self.bindings.pop();
    Ok(())
  }
}
