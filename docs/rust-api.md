---
title: Rust API Reference
description: Complete API reference for MaxPro Rust crate
icon: lucide/gallery-vertical-end
---

# Rust API Reference

This section provides complete documentation for the MaxPro Rust crate.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
maxpro = "0.1"
```

Or via command line:

```bash
cargo add maxpro
```

## Crate Features

| Feature | Description |
|---------|-------------|
| `pyo3-bindings` | Build Python bindings |
| `debug` | Enable plotting utilities |

## Modules

### `enums` - Metrics Enum

```rust
use maxpro::enums::Metrics;
```

Defines available metrics for optimization.

```rust
#[derive(ValueEnum, Clone, Debug)]
pub enum Metrics {
    MaxPro,
    MaxiMin,
}
```

---

## Functions

### generate_lhd

Generates an LHD by taking in a number of samples and a number of dimensions. Creates a non-centered Latin Hypercube Design.

```rust
pub fn generate_lhd(
    n_samples: u64,
    n_dim: u64,
    rng: &mut StdRng
) -> Vec<Vec<f64>>
```

**Arguments:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_samples` | `u64` | Number of samples for the LHD |
| `n_dim` | `u64` | Number of dimensions for the LHD |
| `rng` | `&mut StdRng` | Random number generator |

**Returns:**

`Vec<Vec<f64>>` - A Latin Hypercube Design

**Panics:**

- If `n_samples == 0`
- If `n_dim == 0`
- If `n_samples` or `n_dim` is too large to index

**Example:**

```rust
use maxpro::lhd::generate_lhd;
use rand::{SeedableRng, rngs::StdRng};

let seed = 42u64;
let mut rng = StdRng::seed_from_u64(seed);
let lhd = generate_lhd(100, 5, &mut rng);
```

---

### build_lhd

Using many iterations, selects an LHD that optimizes an allowed metric.

```rust
pub fn build_lhd(
    n_samples: u64,
    n_dim: u64,
    n_iterations: u64,
    metric: Option<enums::Metrics>,
    seed: u64
) -> Vec<Vec<f64>>
```

**Arguments:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_samples` | `u64` | Number of samples |
| `n_dim` | `u64` | Number of dimensions |
| `n_iterations` | `u64` | Number of iterations |
| `metric` | `Option<enums::Metrics>` | Metric to consider (MaxPro or MaxiMin) |
| `seed` | `u64` | Random number seed |

**Returns:**

`Vec<Vec<f64>>` - Latin Hypercube Design that optimizes a metric using random sampling

**Example:**

```rust
use maxpro::{build_lhd, enums::Metrics};

let lhd = build_lhd(
    100,              // n_samples
    10,               // n_dim
    500,              // n_iterations
    Some(Metrics::MaxPro),  // metric
    42                // seed
);
```

---

### maxpro_criterion

Calculates the full, complete MaxPro criterion.

```rust
pub fn maxpro_criterion(design: &Vec<Vec<f64>>) -> f64
```

**Arguments:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `design` | `&Vec<Vec<f64>>` | Design for which to calculate the criterion |

**Returns:**

`f64` - Value of the maximum projection criterion (lower is better)

**Panics:**

- If `design.len() == 0`

**Mathematical Background:**

The MaxPro criterion is:

$$\psi(D) = \left( \frac{2}{n(n-1)} \sum_{i<j} \frac{1}{\prod_{l=1}^{d} (x_{il} - x_{jl})^2} \right)^{1/d}$$

---

### maximin_criterion

Calculates the minimum pairwise distance between points.

```rust
pub fn maximin_criterion(design: &Vec<Vec<f64>>) -> f64
```

**Arguments:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `design` | `&Vec<Vec<f64>>` | Collection of input coordinates |

**Returns:**

`f64` - Minimum pairwise distance between points (higher is better)

**Panics:**

- If `design.len() == 0`

---

### anneal_lhd

Simulated annealing for improving (maximizing or minimizing) a given metric. Supports two annealing strategies: coordinate swap (faster convergence) or jitter (fine-grained exploration).

```rust
pub fn anneal_lhd<F>(
    design: &Vec<Vec<f64>>,
    n_iterations: u64,
    initial_temp: f64,
    cooling_rate: f64,
    metric: F,
    minimize: bool,
    seed: u64,
    swap: bool
) -> Vec<Vec<f64>>
where
    F: Fn(&Vec<Vec<f64>>) -> f64,
```

**Arguments:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `design` | `&Vec<Vec<f64>>` | Initial design of points |
| `n_iterations` | `u64` | Number of iterations for annealing |
| `initial_temp` | `f64` | Initial temperature for Metropolis algorithm |
| `cooling_rate` | `f64` | Cooling rate for annealing |
| `metric` | `F` | A callable function that maps `&Vec<Vec<f64>>` to `f64` |
| `minimize` | `bool` | Whether to minimize or maximize the metric |
| `seed` | `u64` | Random seed |
| `swap` | `bool` | Use coordinate swap annealing (true) or jitter annealing (false) |

**Returns:**

`Vec<Vec<f64>>` - A metric-optimized collection of points (not necessarily a Latin Hypercube)

**Example:**

```rust
use maxpro::anneal::anneal_lhd;
use maxpro::maxpro_utils::maxpro_criterion;

// Coordinate swap annealing (faster convergence)
let optimized = anneal_lhd(
    &lhd,
    5000,      // n_iterations
    1.0,       // initial_temp
    0.99,      // cooling_rate
    maxpro_criterion,
    true,      // minimize
    42,        // seed
    true       // use coordinate swap
);

// Jitter annealing (fine-grained exploration)
let optimized = anneal_lhd(
    &lhd,
    5000,
    1.0,
    0.99,
    maxpro_criterion,
    true,
    42,
    false      // use jitter
);
```

**Recommended Strategy**

A good rule of thumb is to use **20% of iterations for coordinate swap annealing** and **80% for jitter annealing**. **Coordinate swap annealing should always be performed before jitter annealing** - swap first to quickly converge toward a good solution, then use jitter for fine-grained refinement.

For example, with 100,000 total annealing iterations:

```rust
let swap_annealed = anneal_lhd(&lhd, 20000, 1.0, 0.99, metric_fn, true, seed, true);
let final_annealed = anneal_lhd(&swap_annealed, 80000, 1.0, 0.99, metric_fn, true, seed + 1, false);
```

**Important**: It is only possible to achieve state-of-the-art performance using at least some coordinate swap annealing steps. Using jitter annealing alone (without coordinate swap) will not achieve competitive results.

---

### order_design

Reorders a design to optimize the run order for sequential experimentation. The algorithm selects a center point (closest to the design center), then greedily adds remaining points that optimize the chosen criterion at each step. This produces designs where early subsets are already well-distributed.

```rust
pub fn order_design<F>(lhd: Vec<Vec<f64>>, metric: F, minimize: bool) -> Vec<Vec<f64>>
where
    F: Fn(&Vec<Vec<f64>>) -> f64,
```

**Arguments:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `lhd` | `Vec<Vec<f64>>` | Design to reorder |
| `metric` | `F` | A callable function mapping `&Vec<Vec<f64>>` to `f64` |
| `minimize` | `bool` | Whether to minimize the metric (true for MaxPro, false for Maximin) |

**Returns:**

`Vec<Vec<f64>>` - The design with elements reordered for optimal run order

**Algorithm:**

1. Find the point closest to the design center (0.5, 0.5, ...) and place it first
2. For remaining points, greedily select the point that produces the best criterion value when appended
3. Repeat until all points are ordered

**When to Use:**

- Running sequential experiments where early subsets should be well-distributed
- Building surrogate models incrementally and need good coverage from initial runs
- Comparing designs at multiple stopping points (e.g., 10, 25, 50, 100 samples)

**Example:**

```rust
use maxpro::enums::Metrics;
use maxpro::order::order_design;
use maxpro::maxpro_utils::maxpro_criterion;

let lhd = maxpro::build_lhd(100, 5, 500, Some(Metrics::MaxPro), 42);
let ordered = order_design(lhd, maxpro_criterion, true);
```

---

## CLI Usage

Build and run:

```bash
cargo run --release -- [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--iterations <ITERATIONS>` | Number of iterations for optimization |
| `--samples <SAMPLES>` | Number of samples in the design |
| `--ndims <NDIMS>` | Number of dimensions |
| `--metric <METRIC>` | Metric to use: `max-pro` or `maxi-min` |
| `--seed <SEED>` | Random seed for reproducibility |
| `--anneal-iterations <ANNEAL_ITERATIONS>` | Number of annealing iterations (default: 100000) |
| `--anneal-t <ANNEAL_T>` | Initial temperature for annealing (default: 1.0) |
| `--anneal-cooling <ANNEAL_COOLING>` | Cooling rate for annealing (default: 0.99) |
| `--output-path <OUTPUT_PATH>` | Path to save output design |
| `--plot` | Enable plotting (requires debug feature) |

### Examples

Generate MaxPro design:
```bash
cargo run --release -- --iterations 100000 --samples 50 --ndims 2 --metric max-pro
```

Generate Maximin design:
```bash
cargo run --release -- --iterations 100000 --samples 50 --ndims 2 --metric maxi-min
```

---

## Debug Feature

When the `debug` feature is enabled, you can use plotting utilities:

```rust
#[cfg(feature = "debug")]
pub fn plot_x_vs_y(
    data: &Vec<Vec<f64>>,
    output_path: &std::path::Path
) -> Result<(), Box<dyn std::error::Error>>
```

Plots a scatter plot of the first two dimensions of a design for diagnostics.

**Raises:**

- Err if data has fewer than 2 columns or 1 sample

<script id="MathJax-script" async src="https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
  window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };
</script>
