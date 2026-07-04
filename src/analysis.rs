use crate::{
  AstNode, Teachable,
  extract::beam_pareto::{TypeInfo, TypeSet},
  runner::OperationInfo,
};
/// Different analysis methods for the babble language
use egg::{Analysis, DidMerge, EGraph, Language, RecExpr};
use std::{
  collections::{HashMap, HashSet},
  fmt::{Debug, Display},
  hash::{Hash, Hasher},
  marker::PhantomData,
};
/// 实现一个非常简单的Analysis,
/// 包含了类型分析信息，但是此处的类型分析信息更加细节，
/// 同时包含了Vector节点的长度信息
/// 使用了Vec<T>直接存储了enode中可能含有的所有type
/// 这个Analysis只用于LibExtractor和learn
#[derive(Debug, Clone)]
pub struct SimpleAnalysis<Op, T> {
  type_info_map: HashMap<(String, Vec<T>), T>,
  _phantom: PhantomData<Op>,
}
impl<Op, T> Default for SimpleAnalysis<Op, T> {
  fn default() -> Self {
    SimpleAnalysis {
      type_info_map: HashMap::new(),
      _phantom: PhantomData,
    }
  }
}

impl<Op, T> Analysis<AstNode<Op>> for SimpleAnalysis<Op, T>
where
  Op: Clone
    + std::fmt::Debug
    + std::hash::Hash
    + Ord
    + Teachable
    + std::fmt::Display
    + OperationInfo,
  T: Debug + Default + Clone + PartialEq + Ord + Hash,
  AstNode<Op>: TypeInfo<T>,
{
  type Data = TypeSet<T>;
  fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
    let a0 = to.clone();
    to.set.extend(from.set.clone());
    DidMerge(&a0 != to, to != &from)
  }
  fn make(
    egraph: &mut EGraph<AstNode<Op>, Self>,
    enode: &AstNode<Op>,
  ) -> Self::Data {
    let child_types: Vec<T> = enode
      .children()
      .iter()
      .map(|&child| {
        let tys = egraph[child]
          .data
          .set
          .clone()
          .into_iter()
          .collect::<Vec<_>>();
        if tys.len() == 1 {
          tys[0].clone()
        } else {
          let mut merged_ty = tys[0].clone();
          for i in 1..tys.len() {
            merged_ty = AstNode::merge_types(&merged_ty, &tys[i]);
          }
          merged_ty
        }
      })
      .collect();
    let ty = enode.get_rtype(&child_types);
    let mut set = HashSet::new();
    set.insert(ty.clone());
    TypeSet { set }
  }
}
