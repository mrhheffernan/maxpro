---
title: Examples
description: Usage examples for MaxPro library in Python
icon: lucide/code
---

# Examples

This page contains practical examples demonstrating how to use the MaxPro library for generating Latin Hypercube Designs.

## Basic MaxPro Design

This example demonstrates how to generate a semi-optimal MaxPro Latin Hypercube Design and then further optimize it using simulated annealing.

```python
import maxpro


def main():
    # Fix the randomness
    SEED = 42

    # Set number of samples, dimensions, and iterations
    n_samples = 100
    n_dim = 10
    n_iterations = 500

    # Use the maximum projection metric
    metric = "maxpro"

    # Generate a semi-optimal latin hypercube design
    lhd = maxpro.build_lhd(n_samples, n_dim, n_iterations, metric, SEED)
    metric_value = maxpro.maxpro_criterion(lhd)
    print(f"Initial maximum projection metric value: {metric_value}")

    # Use annealing to further minimize the metric value
    n_anneal_iterations = 5000
    initial_temperature = 1
    cooling_rate = 0.99
    minimize = True

    lhd_annealed = maxpro.anneal_lhd(
        lhd,
        n_anneal_iterations,
        initial_temperature,
        cooling_rate,
        metric,
        minimize,
        SEED,
    )

    metric_value_annealed = maxpro.maxpro_criterion(lhd_annealed)
    print(f"Annealed maximum projection metric value: {metric_value_annealed}")


if __name__ == "__main__":
    main()
```

### Key Points

- **Seed**: Set a seed for reproducibility
- **Samples & Dimensions**: Adjust `n_samples` and `n_dim` for your design space
- **Iterations**: More iterations generally find better designs but take longer
- **Annealing**: Simulated annealing can further optimize the design (note: may not preserve LHD properties)

## Basic Maximin Design

This example shows how to generate a Maximin Latin Hypercube Design. Note that for Maximin, we want to **maximize** the minimum distance, so `minimize=False` is used in annealing.

```python
import maxpro


def main():
    # Fix the randomness
    SEED = 42

    # Set number of samples, dimensions, and iterations
    n_samples = 100
    n_dim = 3
    n_iterations = 500

    # Use the maximum projection metric
    metric = "maximin"

    # Generate a semi-optimal latin hypercube design
    lhd = maxpro.build_lhd(n_samples, n_dim, n_iterations, metric, SEED)
    metric_value = maxpro.maximin_criterion(lhd)
    print(f"Initial maximin metric value: {metric_value}")

    # Use annealing to further minimize the metric value
    n_anneal_iterations = 5000
    initial_temperature = 1
    cooling_rate = 0.99
    minimize = False  # Maximize for maximin!

    lhd_annealed = maxpro.anneal_lhd(
        lhd,
        n_anneal_iterations,
        initial_temperature,
        cooling_rate,
        metric,
        minimize,
        SEED,
    )

    metric_value_annealed = maxpro.maximin_criterion(lhd_annealed)
    print(f"Annealed maximin metric value: {metric_value_annealed}")


if __name__ == "__main__":
    main()
```

## Generating Random LHD Without Optimization

If you just need a random Latin Hypercube Design without optimization:

```python
import maxpro

# Generate a random LHD (no metric optimization)
lhd = maxpro.generate_lhd(
    n_samples=50,   # Number of sample points
    n_dim=5,        # Number of dimensions
    seed=42         # Optional seed for reproducibility
)

print(f"Generated LHD with shape: {len(lhd)} x {len(lhd[0])}")
```

## Calculating Metrics on Existing Designs

You can calculate metrics on any design matrix:

```python
import maxpro

# Your custom design (list of lists)
design = [
    [0.1, 0.2, 0.3],
    [0.5, 0.1, 0.9],
    [0.9, 0.8, 0.1],
    # ... more rows
]

# Calculate MaxPro criterion
maxpro_value = maxpro.maxpro_criterion(design)
print(f"MaxPro value: {maxpro_value}")

# Calculate Maximin criterion
maximin_value = maxpro.maximin_criterion(design)
print(f"Maximin value: {maximin_value}")
```

## CLI Usage

### MaxPro Design

Generate from command line:

```bash
cargo run --release -- --iterations 100000 --samples 50 --ndims 2 --metric max-pro
```

### Maximin Design

```bash
cargo run --release -- --iterations 100000 --samples 50 --ndims 2 --metric maxi-min
```

## Parameter Tuning Tips

### For Better Designs

1. **Increase iterations**: More random candidates = better chance of finding optimal
2. **Adjust annealing parameters**:
   - Higher initial temperature = more exploration
   - Lower cooling rate = slower cooling, potentially better results
   - More annealing iterations = more refinement

### Performance Considerations

- Use release mode (`--release`) for production
- Python bindings are pre-compiled and optimized
- Parallel processing is used internally for generating candidate designs
