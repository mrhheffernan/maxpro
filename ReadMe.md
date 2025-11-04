# MaxPro

This is a minimal rust implementation of Latin Hypercube design generation with the Maximum Projection metric. 
It pursues an initial random search for a relatively-optimal candidate and (eventually) will allow for further 
optimization of that candidate in search of a better solution.

Usage of this code should cite both this package as implementation and the MaxPro paper,
```
Joseph, V. R., Gul, E., & Ba, S. (2015). Maximum projection designs for computer experiments. Biometrika, 102(2), 371–380.
```

This implementation was inspired by the comparative lack of lightweight LHD options in Rust as well as no Python 
implementations of MaxPro. By implementing the backend in Rust, a performant and scalable approach for MaxPro design
generation in Python is possible.

## Current capabilities
- Generate a random latin hypercube
- Generate many random latin hypercubes, calculate the maximum projection metric, and return the LHD that minimizes the MaxPro design.
- Using `maturin develop --features pyo3-bindings`, can `import maxpro` and generate optimal MaxPro LHDs in Python directly.

## Planned work
- Implement simulated annealing
- Add testing
- Performance improvements