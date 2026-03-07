# MaxPro

[![Rust build](https://github.com/mrhheffernan/maxpro/actions/workflows/rust-build-test.yml/badge.svg)](https://github.com/mrhheffernan/maxpro/actions/workflows/rust-build-test.yml)
[![Documentation](https://img.shields.io/badge/docs-mrhheffernan.github.io/maxpro-blue)](https://mrhheffernan.github.io/maxpro/)

**[Documentation](https://mrhheffernan.github.io/maxpro/)**

This is a minimal Rust implementation of Latin Hypercube Design (LHD) generation with the Maximum Projection metric. 
It pursues an initial random search for a relatively-optimal candidate and allows for further 
optimization of that candidate (or any other supplied) in search of a better solution, although
this optimization is not guaranteed to preserve the design as a LHD.

The same tools are provided for the Maximin design metric.

Usage of this code should cite both this package as implementation and the MaxPro paper,
```
Joseph, V. R., Gul, E., & Ba, S. (2015). Maximum projection designs for computer experiments. Biometrika, 102(2), 371–380.
```

This implementation was inspired by the comparative lack of lightweight LHD options in Rust as well as no Python 
implementations of MaxPro. Building this in Rust unlocks performance by default.

The Maximin metric is included for additional functionality and performance benchmarking against reference implementations in Python.

## Current capabilities
- Generate a random latin hypercube
- Generate many random latin hypercubes, calculate the maximum projection metric, and return the LHD that minimizes the MaxPro criterion: `cargo run --release -- --iterations 100000 --samples 50 --ndims 2 --metric max-pro`
- Generate many random latin hypercubes, calculate the maximin metric, and return the LHD that maximizes the minimum distance between points: `cargo run --release -- --iterations 100000 --samples 50 --ndims 2 --metric maxi-min`
- Using `maturin develop --release --features pyo3-bindings`, can `import maxpro` and generate optimal MaxPro LHDs in Python directly.
- Perturb a LHD to optimize its metric

## Usage

### Rust
Add the crate to your project via Cargo (`cargo add maxpro`), then `using maxpro::<>` you can use any of the underlying components.

### Python
Install with `pip install maxpro`, `uv add maxpro`, or your other favorite package management tool.

In version 0.1.0, you must import as
```
import _maxpro as maxpro
```
in 0.1.1, this is replaced with
```
import maxpro
```

## Planned work 

Versions 0.1.* are reserved for bug fixes and performance improvements to existing functionality, new features will land in 0.2.0.

### 0.1.1
- Resolve python import syntax
- Make output path optional
- Seedable RNGs (pulled in from 0.2.0)

### 0.2.0
- Ordering the designs for optimal execution order

## AI Policy
This project's AI policy is that no AI-written code is included in the core Rust module or in the python bindings. AI-written code may be present in the `python/` directory but is restricted to analysis. AI code is not used for benchmarking either correctness or speed. 

Gemini code review is used in development and any code suggestions must be human tested. No AI-generated or vibe-coded contributions will be accepted.

## Style
Rust formatting by `cargo fmt`. Python formatting by `ruff`. `flake8` used for PEP, line length not enforced in docstrings within reason.

## Contributing and feature requests
For both contributing and feature requests, please begin by filing an issue specifying either a bug or a feature request. The issues will then be prioritized for inclusion in the next release against other open issues or planned features. To resolve an issue, open a pull request and link it to the issue.

## Benchmarks
MaxPro design and optimization: The Rust implementation usually finds a better metric than the Python one (e.g. 5.95 instead of 7.51) and is ~84x faster; (0.0035s instead of 0.3s for Python on a Macbook Air M2) for 5 samples in 2D across 10,000 iterations. Increasing this to 50 samples in 3D, this implementation's result continues to best the Python result (72.5 vs 91) and is ~1440x faster (0.0178s vs 25.72s)

Maximin design and optimization is benchmarked against PyDOE3 as the reference implementation. The Rust and Python implementations return almost identical results (0.22 in this implementation vs 0.21 in PyDOE3) with this implementation offering a 2.63x speedup for 5 samples in 2D across 10,000 iterations. Increasing this to 50 samples in 3D, this implementation returns a better result than PyDOE3 (0.2204 vs 0.2072) with a 2.9x speedup (0.0286s vs. 0.083s).


Benchmarks are run with `python/comparisons.py`.
Last PR's benchmarks:
```
Rust calculation: 72.39111811323957 in 0.016954898834228516 s
Python calculation: 89.75996904868765 in 25.15124797821045 s
Python / Rust ratio: 1483.4207047838681
Benchmarking Maximin time against reference implementation
Rust criterion 0.22187155920586812 in 0.02622389793395996 s
Python criterion 0.21295020774372542 in 0.07755494117736816 s
Python/Rust ratio: 2.9574146975661644
```

The MaxPro metric calculation can be differentially tested against the R package as the source of truth. 
Current comparisons show agreement up to a relative tolerance of 1e-7.

Design generation can also be benchmarked for speed and metric value. A comparison of metric calculation
is shown below for the same design, as well as design generation with approximately matching parameters.
The R implementation presently generates a more optimal design than this implementation due in part
to this implementation's simplistic annealing approach as of v0.1.1.

Benchmarks below can be reproduced [here](https://colab.research.google.com/drive/1-cgXaP92jp1tPd3w7SQtVEr2WJogOM3u?usp=sharing) with source for local reproduction, including R design generation, in `python/comparison_r.py`.
```
R calculation: 95.98506 
Rust calculation: 95.98505099515629
Design with metric 136.20062378481447 found in 3.3881406784057617 seconds
Rust/R runtime ratio: 0.6156059584849821
```

