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

pub mod ast_node;
pub mod co_occurrence;
mod dfta;
// pub mod dreamcoder;
// pub mod experiments;
pub mod extract;
pub mod learn;
pub mod rewrites;
pub mod sexp;
pub mod simple_lang;
pub mod teachable;
pub mod util;
