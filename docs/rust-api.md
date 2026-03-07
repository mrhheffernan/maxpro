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

Simulated annealing for improving (maximizing or minimizing) a given metric.

```rust
pub fn anneal_lhd<F>(
    design: &Vec<Vec<f64>>,
    n_iterations: u64,
    initial_temp: f64,
    cooling_rate: f64,
    metric: F,
    minimize: bool,
    seed: u64
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

**Returns:**

`Vec<Vec<f64>>` - A metric-optimized collection of points (not necessarily a Latin Hypercube)

**Example:**

```rust
use maxpro::{anneal_lhd, maxpro_utils::maxpro_criterion};

let optimized = anneal_lhd(
    &lhd,
    5000,      // n_iterations
    1.0,       // initial_temp
    0.99,      // cooling_rate
    maxpro_criterion,
    true,      // minimize
    42         // seed
);
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
