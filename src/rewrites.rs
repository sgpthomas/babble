//! Utilities for loading and parsing rewrites files.
//!
//! # Examples
//!
//! A rewrites file might look like this:
//!
//! ```text
//! one_plus_one: (+ 1 1) => 2
//! commutative: (+ ?x ?y) => (+ ?y ?x)
//! ```

use anyhow::anyhow;
use egg::{
  Analysis, Condition, ConditionalApplier, EGraph, FromOp, Id, Language,
  Pattern, Rewrite, Subst, Var,
};
use itertools::Itertools;
use std::{
  collections::HashMap, error::Error, fmt::Debug, fs, io::ErrorKind,
  path::Path, str::FromStr,
};

use crate::extract::beam_pareto::TypeInfo;
/// Returns all the rewrites in the specified file.
///
/// # Errors
/// This function will return an error if the file doesn't exist or can't be
/// opened.
///
/// It will also return an error if it could not parse the file.
pub fn from_file<L, A, P, T>(path: P) -> anyhow::Result<Vec<Rewrite<L, A>>>
where
  L: Language + FromOp + Sync + Send + 'static + TypeInfo<T>,
  A: Analysis<L>,
  <A as Analysis<L>>::Data: PartialEq<T>,
  P: AsRef<Path>,
  L::Error: Send + Sync + Error,
  T: FromStr + Send + Sync + 'static + Debug + Clone,
  <T as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
  let contents = fs::read_to_string(path)?;
  parse(&contents)
}

/// If the file specified by `path` exists, parse the file and return the
/// resulting rewrites. If the file does not exist, return `None`.
///
/// # Errors
/// This function will return an error if the file exists but can't be opened.
///
/// It will also return an error if it could not parse the file.
pub fn try_from_file<L, A, P, T>(
  path: P,
) -> anyhow::Result<Option<Vec<Rewrite<L, A>>>>
where
  L: Language + FromOp + Sync + Send + 'static + TypeInfo<T>,
  A: Analysis<L>,
  <A as Analysis<L>>::Data: PartialEq<T>,
  P: AsRef<Path>,
  L::Error: Send + Sync + Error,
  T: FromStr + Send + Sync + 'static + Debug + Clone,
  <T as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
  Ok(match fs::read_to_string(path) {
    Ok(contents) => Some(parse(&contents)?),
    Err(e) => match e.kind() {
      ErrorKind::NotFound => None,
      _ => Err(e)?,
    },
  })
}

/// Parse a rewrites file.
///
/// # Errors
/// This function will return an error if the rewrites file is invalid.
pub fn parse<L, A, T>(file: &str) -> anyhow::Result<Vec<Rewrite<L, A>>>
where
  L: Language + FromOp + Sync + Send + 'static + TypeInfo<T>,
  A: Analysis<L>,
  <A as Analysis<L>>::Data: PartialEq<T>,
  L::Error: Send + Sync + Error,
  T: FromStr + Send + Sync + 'static + Debug + Clone,
  <T as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
  file
    .lines()
    .filter_map(|line| {
      let line = line.split_once("//").map(|(l, _)| l).unwrap_or(line).trim();
      (!line.is_empty()).then_some(line)
    })
    .map(|line| {
      let (name, (lhs, rhs, ac_flag)) = parse_rewrite_line::<T>(line)?;
      let (lhs_pat, rhs_pat) = (lhs.parse()?, rhs.parse()?);
      let condition = if let Some(cond_str) =
        line.split_once("where").map(|(_, c)| c.trim())
      {
        parse_condition::<T>(cond_str, &lhs_pat)?
      } else {
        TypeMatch::default()
      };
      let rewrites =
        build_rewrites(name, &lhs_pat, &rhs_pat, &condition, ac_flag)?;
      Ok(rewrites)
    })
    .collect::<Result<Vec<_>, _>>()
    .map(|v| v.into_iter().flatten().collect())
}

/// Parse the lhs and rhs of a rewrite rule from a line in the rewrites file.
pub fn parse_lhs_condition<L, T>(
  line: &str,
) -> anyhow::Result<(Pattern<L>, TypeMatch<T>)>
where
  L: Language + FromOp + Sync + Send + 'static + TypeInfo<T>,
  L::Error: Send + Sync + Error,
  T: FromStr + Send + Sync + 'static + Debug + Clone,
  <T as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
  let (_, (lhs, _, _)) = parse_rewrite_line::<T>(line)?;
  let lhs_pat = lhs
    .parse::<Pattern<L>>()
    .map_err(|e| anyhow!("Failed to parse LHS pattern: {}", e))?;
  let condition =
    if let Some(cond_str) = line.split_once("where").map(|(_, c)| c.trim()) {
      parse_condition::<T>(cond_str, &lhs_pat)?
    } else {
      TypeMatch::default()
    };
  Ok((lhs_pat, condition))
}

// 辅助函数 1: 解析重写行
fn parse_rewrite_line<T>(
  line: &str,
) -> anyhow::Result<(&str, (&str, &str, bool))> {
  let (rewrite, condition) = line.split_once("where").unwrap_or((line, ""));
  let ac_flag = rewrite.contains("<==>");
  let split_char = if ac_flag { "<==>" } else { "==>" };

  let (lhs, rhs) = rewrite
    .split_once(split_char)
    .ok_or_else(|| anyhow!("Missing '{}' in: {}", split_char, rewrite))?;
  Ok((line, (lhs.trim(), rhs.trim(), ac_flag)))
}

// 辅助函数 2: 解析条件
fn parse_condition<T>(
  cond_str: &str,
  lhs_pat: &Pattern<impl Language>,
) -> anyhow::Result<TypeMatch<T>>
where
  T: FromStr + Clone + 'static + Send + Debug,
  <T as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
  if cond_str.contains("all") {
    let types = cond_str
      .trim_start_matches("all")
      .trim_matches(|c| c == '(' || c == ')')
      .split(':')
      .nth(1)
      .ok_or_else(|| anyhow!("Invalid all condition format"))?
      .trim();
    let type_list = types
      .strip_prefix('[')
      .and_then(|s| s.strip_suffix(']'))
      .map(|s| s.split('|'))
      .unwrap_or_else(|| types.split('|'))
      .map(|s| s.trim().parse())
      .collect::<Result<Vec<T>, _>>()?;
    Ok(TypeMatch::from_vars(lhs_pat.vars().into_iter(), type_list))
  } else {
    cond_str
      .parse::<TypeMatch<T>>()
      .map_err(|e| anyhow!("Failed to parse condition: {}", e))
  }
}

// 修正后的 build_rewrites 函数
fn build_rewrites<L, A, T>(
  name: &str,
  lhs: &Pattern<L>,
  rhs: &Pattern<L>,
  condition: &TypeMatch<T>,
  ac_flag: bool,
) -> anyhow::Result<Vec<Rewrite<L, A>>>
where
  L: Language + FromOp + Sync + Send + 'static + TypeInfo<T>,
  T: Debug + Clone + Send + Sync + 'static,
  A: Analysis<L>,
  <A as Analysis<L>>::Data: PartialEq<T>,
{
  let mut rewrites = Vec::new();
  // 始终添加正向规则
  if condition.is_empty() {
    rewrites.push(
      Rewrite::new(name, lhs.clone(), rhs.clone())
        .map_err(|e| anyhow!("Rewrite creation failed: {}", e))?,
    );
  } else {
    let conditional_applier = ConditionalApplier {
      condition: condition.clone(),
      applier: rhs.clone(),
    };
    rewrites.push(
      Rewrite::new(name, lhs.clone(), conditional_applier)
        .map_err(|e| anyhow!("Rewrite creation failed: {}", e))?,
    );
  }

  // AC标志为真时添加反向规则
  if ac_flag {
    if condition.is_empty() {
      rewrites.push(
        Rewrite::new(&format!("{}_reverse", name), rhs.clone(), lhs.clone())
          .map_err(|e| anyhow!("Reverse rewrite creation failed: {}", e))?,
      );
    } else {
      let reverse_conditional_applier = ConditionalApplier {
        condition: condition.clone(),
        applier: lhs.clone(),
      };
      rewrites.push(
        Rewrite::new(
          &format!("{}_reverse", name),
          rhs.clone(),
          reverse_conditional_applier,
        )
        .map_err(|e| anyhow!("Reverse rewrite creation failed: {}", e))?,
      );
    }
  }
  Ok(rewrites)
}

// Condition
#[derive(Debug, Clone)]
pub struct TypeMatch<T> {
  pub type_map: HashMap<Var, Vec<T>>,
}

impl<T> TypeMatch<T> {
  pub fn new(type_map: HashMap<Var, Vec<T>>) -> Self {
    TypeMatch { type_map }
  }

  pub fn is_empty(&self) -> bool {
    self.type_map.is_empty()
  }

  pub fn from_vars<I>(vars: I, types: Vec<T>) -> Self
  where
    I: IntoIterator<Item = Var>,
    T: Clone,
  {
    let mut type_map = HashMap::new();
    for var in vars {
      type_map.insert(var, types.clone());
    }
    TypeMatch { type_map }
  }
}

impl<T> Default for TypeMatch<T> {
  fn default() -> Self {
    TypeMatch {
      type_map: HashMap::new(),
    }
  }
}

impl<T: FromStr + Send + Debug> FromStr for TypeMatch<T>
where
  <T as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
  type Err = anyhow::Error;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    // Remove "(" and ")"
    let s = s
      .strip_prefix("(")
      .ok_or(anyhow!("Missing '('"))?
      .strip_suffix(")")
      .ok_or(anyhow!("Missing ')')"))?;
    // Split by commas
    let pairs = s.split(',').collect::<Vec<&str>>();
    let mut type_map = HashMap::new();

    for pair in pairs {
      let parts: Vec<&str> = pair.split(':').collect();
      if parts.len() != 2 {
        return Err(anyhow!("Invalid format"));
      }
      // 如果含有[], 那么说明有多个可能的类型
      let rparts = parts[1].trim();
      if rparts.contains("[") {
        let rparts = rparts
          .strip_prefix("[")
          .ok_or(anyhow!("Missing '['"))?
          .strip_suffix("]")
          .ok_or(anyhow!("Missing ']'"))?;
        let rparts = rparts
          .split('|')
          .map(|s| s.trim().parse::<T>())
          .collect::<Result<Vec<T>, _>>()?;
        println!("rparts: {:?}", rparts);
        type_map.insert(parts[0].trim().parse::<Var>()?, rparts);

        continue;
      } else {
        // Parse Var and  T
        let var = parts[0].trim().parse::<Var>()?;
        let egg_type = vec![parts[1].trim().parse::<T>()?];
        type_map.insert(var, egg_type);
      }
    }
    Ok(TypeMatch { type_map })
  }
}

impl<L, N, T> Condition<L, N> for TypeMatch<T>
where
  L: Language + TypeInfo<T>,
  N: Analysis<L>,
  <N as Analysis<L>>::Data: PartialEq<T> + Debug,
  T: Debug,
{
  fn check(
    &self,
    egraph: &mut EGraph<L, N>,
    _eclass: Id,
    subst: &Subst,
  ) -> bool {
    for (var, egg_type) in &self.type_map {
      if let Some(ecls_id) = subst.get(var.clone()) {
        let ty = &egraph[*ecls_id].data;
        let mut found = false;
        for t in egg_type {
          if ty == t {
            found = true;
            break;
          }
        }
        if !found {
          return false;
        }
      }
    }
    true
  }

  fn vars(&self) -> Vec<Var> {
    let vars = self.type_map.keys().cloned().collect::<Vec<_>>();
    vars
  }
}
