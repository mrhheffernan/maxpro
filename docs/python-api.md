---
title: Python API Reference
description: Complete API reference for MaxPro Python bindings
icon: lucide/braces
---

# Python API Reference

This section provides complete documentation for the MaxPro Python module.

## Installation

```bash
pip install maxpro
```

## Module Functions

### generate_lhd

Generates a random Latin Hypercube Design without optimization.

```python
maxpro.generate_lhd(n_samples: int, n_dim: int, seed: int | None = None) -> list[list[float]]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_samples` | `int` | Number of samples for the LHD. Must be positive and nonzero. |
| `n_dim` | `int` | Number of dimensions for the LHD. Must be positive and nonzero. |
| `seed` | `int`, optional | Seed for the random number generator. If not provided, a random seed is used. |

**Returns:**

`list[list[float]]` - A Latin Hypercube Design as a list of rows.

**Raises:**

- `ValueError` if `n_samples` or `n_dim` is 0
- `ValueError` if `n_samples` or `n_dim` is too large to index

**Example:**

```python
import maxpro

lhd = maxpro.generate_lhd(n_samples=50, n_dim=5, seed=42)
```

---

### build_lhd

Builds an optimal Latin Hypercube Design by searching through multiple random candidates and selecting the one that optimizes the specified metric.

```python
maxpro.build_lhd(
    n_samples: int,
    n_dim: int,
    n_iterations: int,
    metric: str,
    seed: int | None = None
) -> list[list[float]]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_samples` | `int` | Number of samples for the LHD. Must be positive and nonzero. |
| `n_dim` | `int` | Number of dimensions in which to generate points. Must be positive and nonzero. |
| `n_iterations` | `int` | Number of iterations to use to search for an optimal LHD. Must be positive and nonzero. |
| `metric` | `str` | Metric to use. Must be either `'maximin'` or `'maxpro'`. |
| `seed` | `int`, optional | Seed for the random number generator. If not provided, this is randomly chosen. |

**Returns:**

`list[list[float]]` - A semi-optimal Latin Hypercube Design that optimizes the specified metric.

**Raises:**

- `ValueError` if any parameter is 0
- `ValueError` if `n_samples` or `n_dim` is too large to index
- `ValueError` if `metric` is not `'maxpro'` or `'maximin'`

**Example:**

```python
import maxpro

# Generate a MaxPro design
lhd = maxpro.build_lhd(
    n_samples=100,
    n_dim=10,
    n_iterations=500,
    metric="maxpro",
    seed=42
)

# Generate a Maximin design
lhd = maxpro.build_lhd(
    n_samples=100,
    n_dim=3,
    n_iterations=500,
    metric="maximin",
    seed=42
)
```

---

### maxpro_criterion

Calculates the Maximum Projection (MaxPro) criterion for an input design.

```python
maxpro.maxpro_criterion(design: list[list[float]]) -> float
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `design` | `list[list[float]]` | Design of interest. Must not be empty. |

**Returns:**

`float` - Maximum projection criterion value. Lower values indicate better space-filling properties.

**Raises:**

- `ValueError` if design is empty

**Mathematical Background:**

The MaxPro criterion is calculated as:

$$\psi(D) = \left( \frac{2}{n(n-1)} \sum_{i<j} \frac{1}{\prod_{l=1}^{d} (x_{il} - x_{jl})^2} \right)^{1/d}$$

where $n$ is the number of samples and $d$ is the number of dimensions.

**Example:**

```python
import maxpro

design = [[0.1, 0.2], [0.5, 0.1], [0.9, 0.8]]
value = maxpro.maxpro_criterion(design)
print(f"MaxPro value: {value}")  # Lower is better
```

---

### maximin_criterion

Calculates the Maximin criterion for an input design.

```python
maxpro.maximin_criterion(design: list[list[float]]) -> float
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `design` | `list[list[float]]` | Input design. Must not be empty. |

**Returns:**

`float` - Maximin criterion value (minimum pairwise distance). Higher values indicate better spread.

**Raises:**

- `ValueError` if design is empty

**Mathematical Background:**

The Maximin criterion is the minimum Euclidean distance between any pair of points in the design:

$$\text{maximin}(D) = \min_{i < j} \| x_i - x_j \|$$

**Example:**

```python
import maxpro

design = [[0.1, 0.2], [0.5, 0.1], [0.9, 0.8]]
value = maxpro.maximin_criterion(design)
print(f"Maximin value: {value}")  # Higher is better
```

---

### anneal_lhd

Anneals a Latin Hypercube Design to optimize its metric value using simulated annealing. Supports two strategies: coordinate swap (faster convergence, maintains LHD structure better) or jitter (fine-grained exploration).

```python
maxpro.anneal_lhd(
    design: list[list[float]],
    n_iterations: int,
    initial_temp: float,
    cooling_rate: float,
    metric_name: str,
    minimize: bool,
    swap: bool,
    seed: int | None = None
) -> list[list[float]]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `design` | `list[list[float]]` | Input Latin Hypercube Design to optimize |
| `n_iterations` | `int` | Number of optimization iterations. Must be positive and nonzero. |
| `initial_temp` | `float` | Initial temperature for annealing. Must be positive. |
| `cooling_rate` | `float` | Cooling rate for annealing (multiplied by temperature each iteration) |
| `metric_name` | `str` | Metric name; options are `"maxpro"` and `"maximin"` |
| `minimize` | `bool` | Whether to minimize the metric (True for MaxPro, False for Maximin) |
| `swap` | `bool` | Use coordinate swap annealing (True) or jitter annealing (False) |
| `seed` | `int`, optional | Seed for the random number generator |

**Returns:**

`list[list[float]]` - Optimized design (note: may not preserve LHD properties)

**Raises:**

- `ValueError` if `n_iterations` is 0
- `ValueError` if `initial_temp` is not positive
- `ValueError` if `metric_name` is not recognized

**Example:**

```python
import maxpro

# First generate a design
lhd = maxpro.build_lhd(
    n_samples=100,
    n_dim=5,
    n_iterations=500,
    metric="maxpro",
    seed=42
)

# Coordinate swap annealing (faster convergence)
optimized = maxpro.anneal_lhd(
    design=lhd,
    n_iterations=5000,
    initial_temp=1.0,
    cooling_rate=0.99,
    metric_name="maxpro",
    minimize=True,
    swap=True,
    seed=42
)

# Jitter annealing (fine-grained exploration)
optimized = maxpro.anneal_lhd(
    design=lhd,
    n_iterations=5000,
    initial_temp=1.0,
    cooling_rate=0.99,
    metric_name="maxpro",
    minimize=True,
    swap=False,
    seed=42
)
```

**Annealing Tips:**

- Use `swap=True` for faster convergence while preserving LHD structure
- Use `swap=False` (jitter) for fine-grained exploration but may break LHD properties
- `initial_temp`: Higher values allow more exploration but may take longer to converge
- `cooling_rate`: Closer to 1.0 means slower cooling (more refinement)
- `n_iterations`: More iterations = more refinement but slower

**Recommended Strategy**

A good rule of thumb is to use **20% of iterations for coordinate swap annealing** and **80% for jitter annealing**. **Coordinate swap annealing should always be performed before jitter annealing** - swap first to quickly converge toward a good solution, then use jitter for fine-grained refinement.

For example, with 50,000 total annealing iterations:

```python
# 20% swap (10k iterations) - perform this FIRST
swap_annealed = maxpro.anneal_lhd(
    design=lhd,
    n_iterations=10000,
    initial_temp=1.0,
    cooling_rate=0.99,
    metric_name="maxpro",
    minimize=True,
    swap=True,
    seed=42
)

# 80% jitter (40k iterations) - perform this SECOND
optimized = maxpro.anneal_lhd(
    design=swap_annealed,
    n_iterations=40000,
    initial_temp=1.0,
    cooling_rate=0.99,
    metric_name="maxpro",
    minimize=True,
    swap=False,
    seed=43
)
```

**Important**: It is only possible to achieve state-of-the-art performance using at least some coordinate swap annealing steps. Using jitter annealing alone (without coordinate swap) will not achieve competitive results.

---

### order_design

Reorders a design to optimize the run order for sequential experimentation. The algorithm selects a center point (closest to the design center), then greedily adds remaining points that optimize the chosen criterion at each step. This produces designs where early subsets are already well-distributed, making it ideal for surrogate modeling workflows where you want preliminary results from initial runs.

```python
maxpro.order_design(design: list[list[float]], metric_name: str) -> list[list[float]]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `design` | `list[list[float]]` | Design to reorder. Must not be empty. |
| `metric_name` | `str` | Criterion to use: `'maxpro'` or `'maximin'` |

**Returns:**

`list[list[float]]` - The design with elements reordered for optimal run order

**Raises:**

- `ValueError` if design is empty
- `ValueError` if `metric_name` is not `'maxpro'` or `'maximin'`

**Algorithm:**

1. **Center Selection**: Find the point closest to the design center (0.5, 0.5, ...) and place it first
2. **Greedy Selection**: For remaining points, evaluate each candidate by appending it to the current ordered design and computing the criterion value. Select the point that produces the best value.
3. **Repeat** until all points are ordered

**When to Use:**

Use `order_design` when:
- Running sequential experiments where early subsets should be well-distributed
- Building surrogate models incrementally and need good coverage from initial runs
- Comparing designs at multiple stopping points (e.g., 10, 25, 50, 100 samples)

**Example:**

```python
import maxpro

# Generate a design
lhd = maxpro.build_lhd(
    n_samples=100,
    n_dim=5,
    n_iterations=500,
    metric="maxpro",
    seed=42
)

# Order for optimal run order
ordered = maxpro.order_design(lhd, metric_name="maxpro")

# Now you can use subsets like ordered[:10], ordered[:25], etc.
# and each subset will already be well-distributed
```

---

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
