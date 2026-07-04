//! A simple HLS scheduler used for estimating the area and latency of the
//! learned libraries.

use std::{
  cmp,
  collections::{HashMap, HashSet},
  fmt::Debug,
  hash::Hash,
};

use crate::{
  BindingExpr, LibId, Teachable,
  ast_node::AstNode,
  bb_query::{self, BBQuery},
  runner::OperationInfo,
};
use egg::{Language, RecExpr};
use lexpr::print;

/// A trait for languages that support HLS scheduling.
pub trait Schedulable<LA, LD>
where
  Self: Sized,
{
  /// Returns the delay of the operation.
  #[must_use]
  fn op_delay(&self, ld: &LD, args: Vec<Self>) -> usize;

  /// Returns the area of the operation.
  #[must_use]
  fn op_area(&self, la: &LA, args: Vec<Self>) -> usize;

  /// Returns the latency of the operation.
  #[must_use]
  fn op_latency(&self) -> usize;

  /// Returns the latency of the operation in the case of running on cpu
  #[must_use]
  fn op_latency_cpu(&self, bb_query: &BBQuery) -> f64;
}

impl<Op> AstNode<Op>
where
  Op: Clone,
{
  pub fn get_op_args(&self, expr: &RecExpr<Self>) -> Vec<Self> {
    self
      .args()
      .iter()
      .map(|id| {
        let idx: usize = (*id).into();
        expr[idx].clone()
      })
      .collect::<Vec<Self>>()
  }
}

/// A struct representing a scheduler which used to estimate the area and
/// latency of the learned libraries.
#[derive(Clone)]
pub struct Scheduler<LA, LD> {
  /// The clock period of timing scheduling.
  clock_period: usize,
  /// The area estimator.
  area_estimator: LA,
  /// The delay estimator.
  delay_estimator: LD,
  /// BB query for the CPU latency.
  bb_query: BBQuery,
  // Maybe more parameters in the future.
}

/// The result type of scheduling operations, containing (latency_cpu,
/// latency_acc, area)
pub type ScheduleResult = (f64, f64, usize);

impl<LA, LD> Scheduler<LA, LD> {
  pub fn new(
    clock_period: usize,
    area_estimator: LA,
    delay_estimator: LD,
    bb_query: BBQuery,
  ) -> Self {
    Self {
      clock_period,
      area_estimator,
      delay_estimator,
      bb_query,
    }
  }

  pub fn asap_schedule<Op>(&self, expr: &RecExpr<AstNode<Op>>) -> ScheduleResult
  where
    Op: Eq + Hash + Clone + Teachable + Debug + OperationInfo,
    AstNode<Op>: Schedulable<LA, LD> + Language,
  {
    // println!("Scheduling...");
    // Start cycle of each AST node
    // let mut sc: Vec<usize> = vec![0; expr.len()];
    // // Start time in the cycle of each AST node
    // let mut stic: Vec<usize> = vec![0; expr.len()];
    // let mut scheduled: Vec<bool> = vec![false; expr.len()];
    // let mut unscheduled_count = expr.len();

    // // ASAP scheduling
    // let mut cycle = 0;
    // while unscheduled_count > 0 {
    //   for i in 0..expr.len() {
    //     if !scheduled[i] {
    //       let node = &expr[i.into()];
    //       // Check if the operation is ready to be scheduled and calculate
    // the       // earliest time in the cycle the operation can start
    //       let mut ready = true;
    //       let mut earlist_stic = 0;
    //       for &child in node.args() {
    //         let idx: usize = child.into();
    //         if !scheduled[idx] {
    //           ready = false;
    //           break;
    //         } else {
    //           let dep = &expr[idx.into()];
    //           let mut delay =
    //             dep.op_delay(&self.delay_estimator, dep.get_op_args(expr));
    //           let mut latency = dep.op_latency();
    //           if dep.operation().is_mem() {
    //             // latency = dep.op_latency_cpu(&self.bb_query).ceil() as
    // usize;             latency = 0;
    //             delay = (dep.op_latency_cpu(&self.bb_query)
    //               * self.clock_period as f64) as usize;
    //           }
    //           latency += delay / self.clock_period;
    //           delay %= self.clock_period;
    //           let is_sequential = latency > 0;
    //           if is_sequential {
    //             // Check if the dependent operation is ready
    //             if sc[idx] + latency > cycle {
    //               ready = false;
    //               break;
    //             } else if sc[idx] + latency == cycle {
    //               earlist_stic = cmp::max(earlist_stic, delay);
    //             }
    //           } else {
    //             if sc[idx] == cycle {
    //               earlist_stic = cmp::max(earlist_stic, stic[idx] + delay);
    //             }
    //           }
    //         }
    //       }
    //       if ready {
    //         let mut delay =
    //           node.op_delay(&self.delay_estimator, node.get_op_args(expr));
    //         let mut latency = node.op_latency();
    //         if node.operation().is_mem() {
    //           // latency = node.op_latency_cpu(&self.bb_query).ceil() as
    // usize;           latency = 0;
    //           delay = (node.op_latency_cpu(&self.bb_query)
    //             * self.clock_period as f64) as usize;
    //         }
    //         latency += delay / self.clock_period;
    //         delay %= self.clock_period;
    //         let is_sequential = latency > 0;
    //         match is_sequential {
    //           true => {
    //             sc[i] = cycle;
    //             stic[i] = earlist_stic;
    //             scheduled[i] = true;
    //             unscheduled_count -= 1;
    //           }
    //           false => {
    //             if earlist_stic + delay <= self.clock_period {
    //               sc[i] = cycle;
    //               stic[i] = earlist_stic;
    //               scheduled[i] = true;
    //               unscheduled_count -= 1;
    //             }
    //           }
    //         }
    //       }
    //     }
    //   }
    //   cycle += 1;
    // }

    // Identify the loop in the expression
    let root: usize = expr.root().into();
    let nodes = expr.as_ref();
    // println!("Root node: {:?}", nodes[root].operation().get_bbs_info());
    let min_exe_count =
      nodes[root].operation().op_execution_count(&self.bb_query);
    let mut max_exe_count = min_exe_count;
    let mut have_loop = false;
    for node in expr.iter() {
      if node.operation().is_dowhile() {
        have_loop = true;
      }
      let bb_info = node.operation().get_bbs_info();
      if bb_info.len() >= 1 {
        let bb_entry = self.bb_query.get(&bb_info[0]);
        if let Some(bb_entry) = bb_entry {
          max_exe_count =
            max_exe_count.max(bb_entry.execution_count_normalized);
        }
      }
    }
    // println!(
    //   "Min execution count: {}, Max execution count: {}",
    //   min_exe_count, max_exe_count
    // );
    let loop_length = (max_exe_count / min_exe_count).ceil() as usize;

    // Calculate the gain
    // println!("Calculating gain...");
    // println!("Calculating gain...");
    let mut delay_exprs = vec![0; expr.len()];
    let mut delay_accelerator = 0;
    for i in 0..expr.len() {
      let node = &expr[i.into()];
      // let mut delay =
      //   node.op_delay(&self.delay_estimator, node.get_op_args(expr));
      // let mut latency = node.op_latency();
      // if node.operation().is_mem() {
      //   // latency = node.op_latency_cpu(&self.bb_query).ceil() as usize;
      //   latency = 0;
      //   delay = (node.op_latency_cpu(&self.bb_query) * self.clock_period as
      // f64)     as usize;
      // }
      // latency += delay / self.clock_period;
      let delay = if node.operation().is_mem() {
        (node.op_latency_cpu(&self.bb_query) * self.clock_period as f64)
          as usize
      } else {
        node.op_delay(&self.delay_estimator, node.get_op_args(expr))
      };
      // println!("Node: {:?}, Delay: {}", node.operation(), delay);
      let child_delay = node
        .args()
        .iter()
        .map(|id| {
          let idx: usize = (*id).into();
          delay_exprs[idx]
        })
        .max();
      delay_exprs[i] = match child_delay {
        Some(d) => d + delay,
        None => delay,
      };
      delay_accelerator = cmp::max(delay_accelerator, delay_exprs[i]);
    }
    let mut latency_accelerator =
      delay_accelerator as f64 / self.clock_period as f64;
    if have_loop {
      latency_accelerator += (loop_length - 1) as f64;
    }

    let mut latency_cpu_f64 = 0.0;
    for node in expr {
      if node.operation().is_op() {
        let node_latency = node.op_latency_cpu(&self.bb_query);
        if have_loop {
          latency_cpu_f64 += node_latency
            * node.operation().op_execution_count(&self.bb_query) as f64
            / min_exe_count as f64;
        } else {
          latency_cpu_f64 += node_latency;
        }
      }
    }
    // println!(
    //   "Latency accelerator: {}, Latency CPU: {}",
    //   latency_accelerator, latency_cpu_f64
    // );
    // 取上界
    // let latency_cpu = latency_cpu_f64.ceil() as usize;
    // Calculate the area
    // println!("Calculating area...");
    let mut area = 0;
    for node in expr {
      let node_area =
        node.op_area(&self.area_estimator, node.get_op_args(expr));
      // println!("Node: {:?}, Area: {}", &node.operation(), node_area,);
      area += node_area;
    }
    // println!("Number of nodes: {}", expr.len());
    // println!(
    //   "Latency accelerator: {}, Latency CPU: {}, Area: {}",
    //   latency_accelerator, latency_cpu_f64, area
    // );
    if latency_accelerator > latency_cpu_f64 {
      (0.0, 0.0, 0) // If the accelerator is slower, return a large cost
    } else if area == 0 {
      (0.0, 0.0, 0)
    } else {
      (latency_cpu_f64, latency_accelerator, area)
    }
  }
}

/// Calculate the cost of an expression.
pub fn rec_cost<Op, LA, LD>(
  expr: &RecExpr<AstNode<Op>>,
  bb_query: &BBQuery,
  lat_acc_map: HashMap<(usize, Vec<String>), f64>,
) -> (f64, usize)
where
  AstNode<Op>: Schedulable<LA, LD>,
  Op: Teachable + OperationInfo + Clone + Debug,
{
  let mut expr_mut = expr.clone();
  // // Identify the lambda nodes and their children in the expression
  // for (i, node) in expr.iter().enumerate() {
  //   if let Some(BindingExpr::Lambda(_)) = node.as_binding_expr() {
  //     // Recursively remove the lambda node and its children
  //     let mut stack = vec![i];
  //     // Remove the lambda node and its children
  //     while let Some(idx) = stack.pop() {
  //       let cur_node = expr_mut.get_mut(idx).unwrap();
  //       *cur_node.operation_mut() = Op::make_rule_var("Lambda".into());
  //       for child in cur_node.args() {
  //         let child_idx = usize::from(*child);
  //         stack.push(child_idx);
  //       }
  //     }
  //   }
  // }

  let mut used_lib: HashSet<LibId> = HashSet::new();
  let mut cycles = 0.0;
  let mut area: usize = 0;
  let mut cnt = 0;
  for node in expr_mut.iter() {
    let bbs = node.operation().get_bbs_info();
    if bbs.is_empty() {
      continue; // Skip nodes without BB info
    }
    let bb = bbs[0].clone();

    // if !bb.starts_with("naive_cross_product#entry") {
    //   // Skip the naive_point_product entry BB
    //   continue;
    // }
    let exe_count = node.operation().op_execution_count(bb_query);
    // println!("exe_count: {}", exe_count);
    if let Some(BindingExpr::Lib(lid, _, _, _, lat_acc, cost)) =
      node.as_binding_expr()
    {
      let bbs = node.operation().get_bbs_info();
      let lat_acc = if let Some(lat) = lat_acc_map.get(&(lid.0, bbs.clone())) {
        *lat
      } else {
        lat_acc.0
      };

      cycles += lat_acc * exe_count as f64;
      // println!(
      //   "Node: {:?}, Latency: {}, Area: {}",
      //   node.operation(),
      //   lat_acc * exe_count as f64,
      //   cost
      // );
      if used_lib.insert(lid) {
        area += cost;
      }
    } else if node.operation().is_op() {
      cycles += node.op_latency_cpu(bb_query) * exe_count as f64;
      cnt += 1;
      // println!(
      //   "Node: {:?}, Latency: {}",
      //   node.operation(),
      //   node.op_latency_cpu(bb_query)
      // );
    }
  }
  // println!("cnt: {}", cnt);
  (cycles, area)
}

pub fn dump_rec_cost<Op, LA, LD>(
  expr: &RecExpr<AstNode<Op>>,
  bb_query: &BBQuery,
  lat_acc_map: HashMap<(usize, Vec<String>), f64>,
) -> String
where
  AstNode<Op>: Schedulable<LA, LD>,
  Op: Teachable + OperationInfo + Clone + Debug,
{
  let mut used_lib: HashSet<LibId> = HashSet::new();
  let mut cycles = 0.0;
  let mut area: usize = 0;
  let mut lines = Vec::new();

  for (idx, node) in expr.iter().enumerate() {
    let bbs = node.operation().get_bbs_info();
    if bbs.is_empty() {
      continue;
    }
    let exe_count = node.operation().op_execution_count(bb_query);
    if let Some(BindingExpr::Lib(lid, _, _, _, lat_acc, cost)) =
      node.as_binding_expr()
    {
      let lat_acc = lat_acc_map
        .get(&(lid.0, bbs.clone()))
        .copied()
        .unwrap_or(lat_acc.0);
      let contribution = lat_acc * exe_count as f64;
      cycles += contribution;
      let counted_area = if used_lib.insert(lid) {
        area += cost;
        cost
      } else {
        0
      };
      lines.push(format!(
        "node={idx} kind=lib lib={} bbs={:?} exe_count={} lat_acc={} cycles={} area_counted={}",
        lid.0, bbs, exe_count, lat_acc, contribution, counted_area
      ));
    } else if node.operation().is_op() {
      let latency = node.op_latency_cpu(bb_query);
      let contribution = latency * exe_count as f64;
      cycles += contribution;
      lines.push(format!(
        "node={idx} kind=op op={:?} bbs={:?} exe_count={} cpu_lat={} cycles={}",
        node.operation(), bbs, exe_count, latency, contribution
      ));
    }
  }
  lines.push(format!("total cycles={cycles} area={area}"));
  lines.join("\n")
}

pub fn cycles_for_every_function<Op, LA, LD>(
  expr: &RecExpr<AstNode<Op>>,
  bb_query: &BBQuery,
  lat_acc_map: HashMap<(usize, Vec<String>), f64>,
) -> HashMap<String, f64>
where
  AstNode<Op>: Schedulable<LA, LD>,
  Op: Teachable + OperationInfo + Clone + Debug,
{
  let mut func_costs = HashMap::new();
  for node in expr.iter() {
    let bbs = node.operation().get_bbs_info();
    if bbs.is_empty() {
      continue; // Skip nodes without BB info
    }

    let bb = bbs[0].clone();
    let func_name = bb.split('#').next().unwrap_or(&bb).to_string();
    let exe_count = node.operation().op_execution_count(bb_query);
    if let Some(BindingExpr::Lib(lid, _, _, _, lat_acc, _)) =
      node.as_binding_expr()
    {
      let lat_acc = if let Some(lat) = lat_acc_map.get(&(lid.0, bbs.clone())) {
        *lat
      } else {
        lat_acc.0
      };
      let cycles = lat_acc * exe_count;
      func_costs
        .entry(func_name)
        .and_modify(|e| *e += cycles)
        .or_insert(cycles);
    } else if node.operation().is_op() {
      let cycles = node.op_latency_cpu(bb_query) * exe_count;
      func_costs
        .entry(func_name)
        .and_modify(|e| *e += cycles)
        .or_insert(cycles);
    }
  }
  func_costs
}
