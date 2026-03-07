---
title: MaxPro - Latin Hypercube Design Library
description: A minimal Rust implementation of Latin Hypercube Design generation with Maximum Projection and Maximin metrics.
icon: lucide/rocket
---

# MaxPro

A minimal Rust implementation of Latin Hypercube Design (LHD) generation with the Maximum Projection metric. It provides an initial random search for a relatively-optimal candidate and allows for further optimization of that candidate (or any other supplied) in search of a better solution.

The same tools are provided for the Maximin design metric.

## Citation

Usage of this code should cite both this package as implementation and the MaxPro paper:

> Joseph, V. R., Gul, E., & Ba, S. (2015). Maximum projection designs for computer experiments. Biometrika, 102(2), 371–380.

## Features

- **Generate a random Latin Hypercube**: Create random LHDs with configurable samples and dimensions
- **MaxPro Optimization**: Generate many random LHDs and return the one that minimizes the MaxPro criterion
- **Maximin Optimization**: Generate many random LHDs and return the one that maximizes the minimum distance between points
- **Simulated Annealing**: Further optimize any design using simulated annealing
- **Python Bindings**: Use the library directly from Python via PyO3 bindings

## Installation

### Python

```bash
pip install maxpro
```

Or using uv:

```bash
uv add maxpro
```

### Rust

Add the crate to your project via Cargo:

```bash
cargo add maxpro
```

## Quick Start

### Python

```python
import maxpro

# Generate a semi-optimal MaxPro latin hypercube design
lhd = maxpro.build_lhd(
    n_samples=100,
    n_dim=10,
    n_iterations=500,
    metric="maxpro",
    seed=42
)

# Calculate the MaxPro criterion
metric_value = maxpro.maxpro_criterion(lhd)
print(f"MaxPro metric value: {metric_value}")
```

### Rust

```rust
use maxpro::{build_lhd, enums::Metrics};

fn main() {
    let lhd = build_lhd(
        100,      // n_samples
        10,       // n_dim
        500,      // n_iterations
        Some(Metrics::MaxPro),
        42,       // seed
    );
}
```

## Available Metrics

### MaxPro (Maximum Projection)

The Maximum Projection criterion aims to maximize the minimum projection distance between all pairs of points. This metric is particularly useful for space-filling designs in computer experiments.

### Maximin

The Maximin criterion maximizes the minimum distance between any two points in the design. This ensures good spread across the design space.

## Benchmarks

### MaxPro

- The Rust implementation usually finds a better metric than Python alternatives (e.g., 5.95 instead of 7.51)
- ~84x faster than Python on Macbook Air M2 for 5 samples in 2D across 10,000 iterations
- ~1440x faster for 50 samples in 3D

### Maximin

- Returns almost identical results to PyDOE3 reference implementation
- ~2.9x speedup for 50 samples in 3D

## R Comparison

The MaxPro metric calculation is validated against the R package as the source of truth. Current comparisons show agreement up to a relative tolerance of **1e-7**.

Design generation can also be benchmarked for speed and metric value. A comparison of metric calculation is shown below for the same design, as well as design generation with approximately matching parameters:

| Metric | R | Rust |
|--------|------|------|
| MaxPro criterion | 95.98506 | 95.98505099515629 |

The Rust implementation is **~1.6x faster** than R for design generation. Note that the R implementation may generate more optimal designs due to its more sophisticated optimization approach compared to this implementation's simplistic annealing approach.

Benchmarks can be reproduced using the [Google Colab notebook](https://colab.research.google.com/drive/1-cgXaP92jp1tPd3w7SQtVEr2WJogOM3u) or locally using `python/comparison_r.py`.

## CLI Usage

Generate an optimal MaxPro LHD:

```bash
cargo run --release -- --iterations 100000 --samples 50 --ndims 2 --metric max-pro
```

Generate an optimal Maximin LHD:

```bash
cargo run --release -- --iterations 100000 --samples 50 --ndims 2 --metric maxi-min
```

## Documentation Sections

- [Examples](examples.md) - Usage examples in Python
- [Python API](python-api.md) - Python bindings reference
- [Rust API](rust-api.md) - Rust crate documentation

## Contributing

For contributing and feature requests, please begin by filing an issue specifying either a bug or a feature request. To resolve an issue, open a pull request and link it to the issue.

## AI Policy

No AI-written code is included in the core Rust module or in the Python bindings. AI-written code may be present in the `python/` directory but is restricted to analysis.
