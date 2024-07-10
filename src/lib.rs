//! Library learning using [anti-unification] of e-graphs.
//!
//! [anti-unification]: https://en.wikipedia.org/wiki/Anti-unification_(computer_science)

#![warn(
    clippy::all,
    clippy::pedantic,
    anonymous_parameters,
    elided_lifetimes_in_paths,
    missing_copy_implementations,
    trivial_casts,
    unreachable_pub,
    unused_lifetimes
)]
#![allow(clippy::non_ascii_literal)]
#![allow(clippy::non_canonical_partial_ord_impl)]

mod ast_node;
mod co_occurrence;
mod dfta;
pub mod extract;
mod learn;
pub mod rewrites;
pub mod sexp;
pub mod simple_lang;
mod teachable;
pub mod util;

pub use ast_node::{
    combine_exprs, Arity, AstNode, Expr, PartialExpr, Precedence, Pretty, Printable, Printer,
};
pub use co_occurrence::{COBuilder, CoOccurrences};
pub use learn::{LearnedLibrary, LibId, ParseLibIdError};
pub use teachable::{BindingExpr, DeBruijnIndex, ParseDeBruijnIndexError, Teachable};
