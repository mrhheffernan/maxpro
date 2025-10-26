# MaxPro

This is a minimal rust implementation of Latin Hypercube design generation with the Maximum Projection metric.

Usage of this code should cite both this package as implementation and the MaxPro paper,
```
Joseph, V. R., Gul, E., & Ba, S. (2015). Maximum projection designs for computer experiments. Biometrika, 102(2), 371–380.
```

## Current status
- Generate a random latin hypercube
- Generate many random latin hypercubes, calculate the maximum projection metric, and return the LHD that minimizes the MaxPro design.

## Planned work
- Implement simulated annealing
- Create a Python frontend via PyO3
- Add testing
- Compare calculations between this work and the MaxPro R implementation to ensure metrics match between designs
- Performance improvements