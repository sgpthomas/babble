use std::collections::{BTreeMap, HashMap};

/// this file is used to filter the legal AU
use serde::{Deserialize, Serialize};

/// Configuration for Packed Access strategy
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct PackedAccessConfig {
  /// Whether this strategy is enabled
  pub enabled: bool,
  /// Maximum total bit-width allowed for packed fields
  pub max_packed_width: u32,
  /// List of allowed sub-field widths
  pub pack_sizes: Vec<u32>,
}

impl Default for PackedAccessConfig {
  fn default() -> Self {
    Self {
      enabled: true,
      max_packed_width: 32,
      pack_sizes: vec![8, 16],
    }
  }
}

/// Configuration for Register Grouping strategy
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub struct RegisterGroupingConfig {
  pub enabled: bool,
  /// Number of bits for group selector (hand)
  pub hand_bits: u8,
  /// Number of bits for intra-group displacement
  pub dis_bits: u8,
  /// Maximum number of groups = 2^hand_bits
  pub max_hands: u32,
  /// Allow per-input overrides of the group
  pub allow_override: bool,
  /// Bits consumed by each override
  pub override_bits: u8,
}

impl Default for RegisterGroupingConfig {
  fn default() -> Self {
    Self {
      enabled: false,
      hand_bits: 2,
      dis_bits: 3,
      max_hands: 4,
      allow_override: false,
      override_bits: 2,
    }
  }
}

/// Configuration for Register Neighboring strategy
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub struct RegisterNeighboringConfig {
  pub enabled: bool,
  /// Extra bits consumed when neighbor-enabled
  pub neighbor_extra_bits: u8,
  /// How many adjacent registers can be brought in
  pub max_neighbors: u8,
}

impl Default for RegisterNeighboringConfig {
  fn default() -> Self {
    Self {
      enabled: false,
      neighbor_extra_bits: 1,
      max_neighbors: 1,
    }
  }
}

/// Configuration for cost estimation to guide AU-search
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct EncodingCostModelConfig {
  /// Weight cost per header bit consumed
  pub cost_per_header_bit: f32,
  /// Weight cost per displacement bit consumed
  pub cost_per_dis_bit: f32,
  /// Discount or bonus factor for using packed access
  pub cost_per_packed: f32,
  /// Penalty cost for each neighbor-enabled input/output
  pub cost_per_neighbor: f32,
  /// Penalty cost for each override usage
  pub cost_per_override: f32,
}

impl Default for EncodingCostModelConfig {
  fn default() -> Self {
    Self {
      cost_per_header_bit: 0.1,
      cost_per_dis_bit: 0.2,
      cost_per_packed: -0.5,
      cost_per_neighbor: 0.3,
      cost_per_override: 0.4,
    }
  }
}

/// Top-level config holding all strategies
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CiEncodingConfig {
  pub max_total_bits: u32,
  pub default_bits_per_register: u32,
  pub packed_access: PackedAccessConfig,
  pub register_grouping: RegisterGroupingConfig,
  pub register_neighboring: RegisterNeighboringConfig,
}

impl Default for CiEncodingConfig {
  fn default() -> Self {
    Self {
      max_total_bits: 32,
      default_bits_per_register: 5,
      packed_access: PackedAccessConfig::default(),
      register_grouping: RegisterGroupingConfig::default(),
      register_neighboring: RegisterNeighboringConfig::default(),
    }
  }
}

/// a trait to get the length and the width of a type
pub trait TypeAnalysis {
  /// get the width of the type, if it is not defined, return None
  fn width(&self) -> Option<u32> {
    None
  }
  /// get the length of the type, if it is not defined, return 1
  fn length(&self) -> u32 {
    1
  }
  /// whether the type is a state_type
  fn is_state_type(&self) -> bool {
    false
  }
  /// the extra vector bits
  fn extra_vector_bits(&self) -> u8 {
    2
  }
}

/// this function is used to calculate the maximum number of bits according
/// the config
pub fn max_inputs(config: &CiEncodingConfig) -> u32 {
  let mut count = config.max_total_bits;
  if config.register_grouping.enabled {
    count /= config.register_grouping.dis_bits as u32;
  } else {
    count /= config.default_bits_per_register;
  };

  if config.register_neighboring.enabled {
    count *= (config.register_neighboring.max_neighbors + 1) as u32;
  };

  if config.packed_access.enabled {
    let mini_width = config.packed_access.pack_sizes.iter().min().unwrap_or(&8);
    count *= (config.packed_access.max_packed_width / *mini_width).max(1);
  }
  count
}

/// this function is used to use greedy algorithm to perform packed access
fn greedy_packed_access(
  widths_map: BTreeMap<u32, u32>,
  max_packed_width: u32,
) -> u32 {
  let mut bins = 0;
  let mut count = widths_map.clone();
  while count.values().any(|&v| v > 0) {
    let mut space = max_packed_width;
    for (&w, c) in count.iter_mut().rev() {
      while *c > 0 && w <= space {
        space -= w;
        *c -= 1;
      }
    }

    bins += 1;
  }
  bins
}

/// Filters whether a vector of reg_types can be encoded under the CI
/// constraints Returns true if total bit usage <= config.max_total_bits
pub fn io_filter<T: TypeAnalysis>(
  config: &CiEncodingConfig,
  reg_types: &[T],
) -> bool {
  // if the total inputs exceeds the maximum allowed inputs, we can just return
  // false
  if reg_types.len() as u32 > max_inputs(config) {
    // println!(
    //   "Too many inputs: {} > {}",
    //   reg_types.len(),
    //   max_inputs(config)
    // );
    return false;
  }
  // if the total bits satisfies the max_total_bits, we can just return true
  let total_bits: u32 = reg_types.iter().map(|t| t.width().unwrap_or(32)).sum();
  if total_bits <= config.max_total_bits {
    return true;
  }

  let mut extra_bits = config.max_total_bits;

  let mut vec_extra_bits = 0;
  // calculate the extra bits needed for vector types
  for t in reg_types {
    if t.length() > 1 {
      vec_extra_bits += t.extra_vector_bits() as u32;
    }
  }

  // first check whether we can just use the packed access, this phase we need
  // to pay attention to the detailed width
  let mut register_num = reg_types.len() as u32;
  // in this phase , we may update the register_num
  if config.packed_access.enabled {
    // classify the input types according to their width
    let mut width_count = HashMap::new();
    for t in reg_types {
      if let Some(width) = t.width() {
        *width_count.entry(width).or_insert(0) += 1;
      } else {
        // default to 32 bits if width is not defined
        *width_count.entry(32).or_insert(0) += 1;
      }
    }
    register_num = 0;
    let mut widths_map: BTreeMap<u32, u32> = BTreeMap::new();
    for (width, count) in &width_count {
      if !&config.packed_access.pack_sizes.contains(width) {
        register_num += count;
      } else {
        widths_map.insert(*width, *count);
      }
    }
    // now we can use the greedy algorithm to perform packed access
    let packed_bins =
      greedy_packed_access(widths_map, config.packed_access.max_packed_width);
    register_num += packed_bins;
  }
  if register_num * config.default_bits_per_register + vec_extra_bits
    <= config.max_total_bits
  {
    return true;
  }
  // now we can check whether the register neighboring is enabled
  // From now on , we will just take rigister_num into account
  if config.register_neighboring.enabled {
    // calculate the register number after considering the neighboring
    let neighbor_bits = config.register_neighboring.neighbor_extra_bits as u32;
    let max_neighbors = config.register_neighboring.max_neighbors as u32;
    extra_bits -= neighbor_bits;
    register_num = (register_num + max_neighbors) / (max_neighbors + 1);
  }
  if register_num * config.default_bits_per_register <= extra_bits {
    return true;
  }

  // now we can check whether the register grouping is enabled
  if config.register_grouping.enabled {
    // TODO:Now I do not consider the override bits, just use the hand_bits
    let hand_bits = config.register_grouping.hand_bits as u32;
    let dis_bits = config.register_grouping.dis_bits as u32;
    extra_bits -= hand_bits;
    if register_num * dis_bits + vec_extra_bits <= extra_bits {
      return true;
    }
  }
  false
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Dummy type for testing, with a specified width
  #[derive(Debug, Clone)]
  struct Dummy {
    w: u32,
  }
  impl TypeAnalysis for Dummy {
    fn length(&self) -> u32 {
      1
    }
    fn width(&self) -> Option<u32> {
      Some(self.w)
    }
  }

  #[test]
  fn test_simple_within_limit() {
    let cfg = CiEncodingConfig::default();
    let inputs = vec![Dummy { w: 8 }, Dummy { w: 16 }];
    assert!(io_filter(&cfg, &inputs));
  }

  #[test]
  fn test_exceed_without_packing() {
    let mut cfg = CiEncodingConfig::default();
    cfg.packed_access.enabled = true;
    let dummy = Dummy { w: 20 };
    let inputs = vec![dummy.clone(); 20];
    // total_bits = 40 > 32 and packing disabled
    assert!(!io_filter(&cfg, &inputs));
  }

  #[test]
  fn test_packing_reduces_registers() {
    let cfg = CiEncodingConfig::default();
    // Five 8-bit values can be packed into two registers
    let inputs = vec![Dummy { w: 8 }; 5];
    assert!(io_filter(&cfg, &inputs));
  }

  #[test]
  fn test_neighboring_effect() {
    let mut cfg = CiEncodingConfig::default();
    cfg.register_neighboring.enabled = true;
    let inputs = vec![Dummy { w: 32 }; 2];
    // Two 32-bit inputs with neighboring: still needs two registers, but
    // neighbor extra bit applies Default_bits_per_register=5, so 2*5=10
    // bits + neighbor_extra_bits(1)=11 <=32
    assert!(io_filter(&cfg, &inputs));
  }

  #[test]
  fn test_grouping_effect() {
    let mut cfg = CiEncodingConfig::default();
    cfg.register_grouping.enabled = true;
    // Six inputs requiring grouping: dis_bits=3, hand_bits=2 =>
    // bits_needed=2+6*3=20 <=32
    let inputs = vec![Dummy { w: 8 }; 6];
    assert!(io_filter(&cfg, &inputs));
  }
  #[test]
  fn test_mixed_pack_sizes() {
    let cfg = CiEncodingConfig::default();
    let inputs = vec![
      Dummy { w: 8 },
      Dummy { w: 16 },
      Dummy { w: 8 },
      Dummy { w: 16 },
      Dummy { w: 8 },
      Dummy { w: 16 },
    ];
    // should fit within packed access scheme
    assert!(io_filter(&cfg, &inputs));
  }
}
