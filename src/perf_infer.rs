use egg::{EGraph, Id, RecExpr};
use std::{
  collections::{HashMap, HashSet},
  fmt::Debug,
  hash::Hash,
};

use crate::{
  Arity, AstNode, Teachable,
  bb_query::BBInfo,
  extract::beam_pareto::{ISAXAnalysis, TypeInfo},
  runner::OperationInfo,
};

/// Starting from each root ID, traverse downward through all reachable nodes in
/// the e-graph and propagate the root’s BB info to every enode. Guarantees that
/// no enode’s bbs_info remains empty.
pub fn perf_infer<Op, T>(
  egraph: &mut EGraph<AstNode<Op>, ISAXAnalysis<Op, T>>,
  roots: &[Id],
) where
  Op: Clone
    + Debug
    + Ord
    + Hash
    + Teachable
    + Arity
    + OperationInfo
    + Send
    + Sync
    + BBInfo
    + 'static,
  T: Debug + Default + Clone + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  // 1. We'll track which eclasses we’ve visited already
  let mut visited = HashSet::new();
  // 2. For each eclass ID, we’ll store the chosen BB info that should be
  //    written into that class
  let mut bb_info_map: HashMap<Id, Vec<String>> = HashMap::new();
  // 3. Our manual DFS stack holds (eclass_id, inherited_bb_info)
  let mut work_stack: Vec<(Id, Vec<String>)> = Vec::new();

  // ── Step 1: For each root ID, grab its non-empty BB info directly from its
  // eclass ──
  //
  // We assume each root has at least one enode whose get_mut_bbs_info() is
  // non-empty. That becomes the starting BB info for that root.
  for &root_id in roots {
    let ecls = &mut egraph[root_id];
    let root_bb: Vec<String> = ecls.data.bb.clone();
    // if root_bb.is_empty() {
    //   println!(
    //     "Warning! Root eclass {} has no BB info. This should not happen.",
    //     root_id
    //   );
    // }
    // By assumption, the root always has at least one enode with BB info.
    // Push (root ID, its BB info) onto the stack
    work_stack.push((root_id, root_bb));
  }

  // ── Step 2: DFS — pop (current_id, inherited_bb) and propagate downward ──
  while let Some((current_id, inherited_bb)) = work_stack.pop() {
    // Skip if we’ve already processed this eclass
    if !visited.insert(current_id) {
      continue;
    }

    // Examine the eclass to see if any enode in it has its own BB info. If so,
    // use that instead.
    let mut final_bb = inherited_bb.clone();
    {
      let ecls = &mut egraph[current_id];
      for node in &mut ecls.nodes {
        let bbs = node.operation_mut().get_mut_bbs_info();
        if !bbs.is_empty() {
          // This enode already has its own BB info; override inherited_bb with
          // it
          final_bb = bbs.clone();
          break;
        }
      }
    }

    // If that BB info vector has more than one entry, truncate to just the
    // first.
    if final_bb.len() > 1 {
      final_bb.truncate(1);
    }

    // Record that this eclass (current_id) should end up with final_bb
    bb_info_map.insert(current_id, final_bb.clone());

    // Now push every child eclass onto the stack, passing along final_bb
    {
      let ecls = &mut egraph[current_id];
      for node in &mut ecls.nodes {
        for &child_id in node.args() {
          if !visited.contains(&child_id) {
            work_stack.push((child_id, final_bb.clone()));
          }
        }
      }
    }
  }

  // ── Step 3: Write the collected BB info back into every eclass and every
  // enode within that class ──
  for ecls in egraph.classes_mut() {
    let id = ecls.id;
    if let Some(ecls_bb) = bb_info_map.get(&id) {
      // 3a. Update the eclass’s own analysis data
      ecls.data.bb = ecls_bb.clone();

      // 3b. For each enode in this eclass: if its bbs_info is empty, fill it in
      for node in &mut ecls.nodes {
        let bbs = node.operation_mut().get_mut_bbs_info();
        if bbs.is_empty() {
          *bbs = ecls_bb.clone();
        }
      }
    }
  }

  // ── Step 4: Rebuild to maintain e-graph invariants ──
  egraph.rebuild();
}

pub fn expr_perf_infer<Op>(expr: &mut RecExpr<AstNode<Op>>)
where
  Op: Clone
    + Debug
    + Ord
    + Hash
    + Teachable
    + Arity
    + OperationInfo
    + Send
    + Sync
    + BBInfo
    + 'static,
{
  let mut nodes_without_bbs: Vec<usize> = Vec::new();
  let expr_clone = expr.clone();
  let nodes = expr.as_mut();
  for (i, node) in expr_clone.iter().enumerate() {
    if node.operation().get_bbs_info().is_empty() {
      nodes_without_bbs.push(i);
    } else {
      let bbs = node.operation().get_bbs_info();
      for node_without_bbs in nodes_without_bbs.iter() {
        nodes[*node_without_bbs]
          .operation_mut()
          .get_mut_bbs_info()
          .extend(bbs.iter().cloned());
      }
      nodes_without_bbs.clear();
    }
  }
}
