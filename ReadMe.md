# MaxPro

[![Rust build](https://github.com/mrhheffernan/maxpro/actions/workflows/rust-build-test.yml/badge.svg)](https://github.com/mrhheffernan/maxpro/actions/workflows/rust-build-test.yml)

This is a minimal rust implementation of Latin Hypercube design generation with the Maximum Projection metric. 
It pursues an initial random search for a relatively-optimal candidate and allows for further 
optimization of that candidate (or any other supplied) in search of a better solution.

Usage of this code should cite both this package as implementation and the MaxPro paper,
```
Joseph, V. R., Gul, E., & Ba, S. (2015). Maximum projection designs for computer experiments. Biometrika, 102(2), 371–380.
```

This implementation was inspired by the comparative lack of lightweight LHD options in Rust as well as no Python 
implementations of MaxPro. By implementing the backend in Rust, a performant and scalable approach for MaxPro design
generation in Python is possible.

This implementation also includes the often-less-optimal, but more standard, Maximin metric for additional functionality and benchmarking against reference implementations in Python.

## Current capabilities
- Generate a random latin hypercube
- Generate many random latin hypercubes, calculate the maximum projection metric, and return the LHD that minimizes the MaxPro criterion: `cargo run --release -- --iterations 100000 --samples 50 --ndims 2 --metric max-pro`
- Generate many random latin hypercubes, calculate the maximin metric, and return the LHD that maximizes the minimum distance between points: `cargo run --release -- --iterations 100000 --samples 50 --ndims 2 --metric maxi-min`
- Using `maturin develop --release --features pyo3-bindings`, can `import maxpro` and generate optimal MaxPro LHDs in Python directly.
- Perturb a LHD to optimize its metric

## Planned work 
### 0.1.0
- Add testing
- Documentation for PyO3 bindings
- Crates.io and PyPI deployment
- Comparisons to R implementation for further metric, performance benchmarking and validation.

### 0.2.0
- Ordering the designs for optimal execution order

## AI Policy
This project's AI policy is that no AI-written code is included in the core Rust module or in the python bindings. AI-written code may be present in the `python/` directory but is restricted to analysis. AI code is not used for benchmarking either correctness or speed. 

Gemini code review is used in development and any code suggestions must be human tested.

## Style
Rust formatting by `cargo fmt`. Python formatting by `ruff`. `flake8` used for PEP, line length not enforced in docstrings within reason.

## Contributing and feature requests
For both contributing and feature requests, please begin by filing an issue specifying either a bug or a feature request. The issues will then be prioritized for inclusion in the next release against other open issues or planned features. To resolve an issue, open a pull request and link it to the issue.