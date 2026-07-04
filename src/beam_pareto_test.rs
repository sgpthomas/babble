use std::collections::HashMap;
use std::sync::Arc;

use crate::ast_node::{AstNode, Expr, Pretty};
use crate::extract::beam_pareto::ISAXAnalysis;
use crate::runner::{
  BabbleParetoRunner, LiblearnConfig, ParetoConfig, ParetoRunner, VectorConfig,
};
use crate::simple_lang::{SimpleOp, SimpleType};
use egg::{RecExpr, Rewrite};

// (add (add ?x ?x) (add ?y ?y))
fn create_expr(x: i64, y: i64) -> Expr<SimpleOp> {
  // Create symbols
  let plus = SimpleOp::Add;
  let x = SimpleOp::Const(x);
  let y = SimpleOp::Const(y);

  // Create the expression as a RecExpr
  let mut rec_expr = RecExpr::default();

  // Add nodes to the RecExpr
  let x_id = rec_expr.add(AstNode::new(x, vec![]));
  let y_id = rec_expr.add(AstNode::new(y, vec![]));
  let mul_x_x_id = rec_expr.add(AstNode::new(plus.clone(), vec![x_id, x_id]));
  let mul_y_y_id = rec_expr.add(AstNode::new(plus.clone(), vec![y_id, y_id]));

  rec_expr.add(AstNode::new(plus, vec![mul_x_x_id, mul_y_y_id]));

  Expr::from(rec_expr)
}

#[test]
fn beam_pareto_test() {
  let config = ParetoConfig {
    final_beams: 5,
    inter_beams: 3,
    lib_iter_limit: 1,
    lps: 2,
    learn_constants: false,
    max_arity: None,
    strategy: 0.8,
    clock_period: 10,
    add_all_types: false,
    liblearn_config: LiblearnConfig::default(),
    vectorize_config: VectorConfig::default(),
  };

  let exprs = vec![create_expr(1, 2), create_expr(3, 4)];
  for expr in exprs.clone() {
    println!("{}", Pretty::new(Arc::new(expr)));
  }

  let dsrs: Vec<
    Rewrite<AstNode<SimpleOp>, ISAXAnalysis<SimpleOp, SimpleType>>,
  > = vec![];
  let lib_rewrites = HashMap::new();

  let runner = ParetoRunner::new(dsrs, lib_rewrites, config);
  let result = runner.run_multi(exprs);
  println!("Final cost: {:#?}", result.final_cost);
  // println!("Compression ratio: {:.2}", result.compression_ratio());
  // println!("Number of libraries learned: {}", result.num_libs);
  // println!("Run time: {:?}", result.run_time);
  println!("Final expression: ");
  println!("{}", Pretty::new(Arc::new(result.final_expr.clone())));
}
