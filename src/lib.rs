//! Library learning using [anti-unification] of e-graphs.
//!
//! [anti-unification]: https://en.wikipedia.org/wiki/Anti-unification_(computer_science)

#![warn(
  clippy::all,
  clippy::pedantic,
  anonymous_parameters,
  elided_lifetimes_in_paths,
  missing_copy_implementations,
  // trivial_casts,
  unreachable_pub,
  unused_lifetimes
)]
#![allow(clippy::non_ascii_literal)]
#![allow(clippy::non_canonical_partial_ord_impl)]

pub mod analysis;
pub mod ast_node;
pub mod au_filter;
pub mod au_merge;
pub mod bb_query;
// #[cfg(test)]
// mod beam_pareto_test;
mod co_occurrence;
mod dfta;
pub mod extract;
pub mod learn;
pub mod perf_infer;
pub mod rewrites;
pub mod runner;
pub mod schedule;
pub mod sexp;
// pub mod simple_lang;
pub mod expand;
pub mod teachable;
pub mod util;
pub mod vectorize;

pub use ast_node::{
  Arity, AstNode, Expr, PartialExpr, Precedence, Pretty, Printable, Printer,
  combine_exprs,
};
pub use co_occurrence::{COBuilder, CoOccurrences};
pub use learn::{
  DiscriminantEq, LearnedLibrary, LearnedLibraryBuilder, LibId, ParseLibIdError,
};
pub use runner::{ParetoConfig, ParetoRunner};
pub use teachable::{
  BindingExpr, DeBruijnIndex, ParseDeBruijnIndexError, Teachable,
};
