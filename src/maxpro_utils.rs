use crate::lhd::generate_lhd;
#[cfg(feature = "pyo3-bindings")]
use pyo3::PyResult;
#[cfg(feature = "pyo3-bindings")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;
use rayon::prelude::*;

/// Helper function to calculate the internal sum term of the MaxPro criterion (psi(D)).
/// This sum is the term that is directly minimized in the optimization
/// process.
/// The minimized term is: sum_{i<j} [ 1 / product_{l=1}^{d} (x_il - x_jl)^2 ]
///
/// Arguments:
///     design (&Vec<Vec<f64>>): Design for which to calculate the metric
///
/// Returns:
///     f64: Maxpro inner sum value
///
/// Panics:
///     design.len() == 0: Design cannot be empty
fn maxpro_sum(design: &Vec<Vec<f64>>) -> f64 {
    let n: usize = design.len();
    assert!(n > 0, "Cannot pass an empty design");
    if n < 2 {
        return 0.0;
    }

    let mut inverse_product_sum: f64 = 0.0;
    let epsilon: f64 = 1e-12; // Small constant to prevent division by zero

    for i in 0..n {
        for j in (i + 1)..n {
            // Calculate the product term: product_{l=1}^{d} (x_il - x_jl)^2
            let row_i: &Vec<f64> = &design[i];
            let row_j: &Vec<f64> = &design[j];
            let mut product_of_squared_diffs: f64 = 1.0;

            // Use zip to iterate over both rows simultaneously
            for (x_i_l, x_j_l) in row_i.iter().zip(row_j.iter()) {
                let diff: f64 = x_i_l - x_j_l;
                let diff_sq: f64 = diff * diff;
                product_of_squared_diffs *= diff_sq;
            }

            // Add epsilon to prevent division by zero
            product_of_squared_diffs += epsilon;

            // // Sum the inverse products
            inverse_product_sum += 1.0 / product_of_squared_diffs;
        }
    }

    inverse_product_sum
}

/// Calculates the full, complete MaxPro criterion
///
/// Arguments:
///     design (&Vec<Vec<f64>>): Design for which to calculate the criterion
///
/// Returns:
///     f64: Value of the maximum projection criterion
///
/// Panics:
///     design.len() == 0: Cannot pass an empty design
pub fn maxpro_criterion(design: &Vec<Vec<f64>>) -> f64 {
    let n: usize = design.len();
    assert!(n > 0, "Cannot pass an empty design");
    if n < 2 {
        return 0.0;
    }
    let d: usize = design[0].len();
    assert!(d > 0, "Must have finite, positive number of dimensions");
    let inverse_product_sum: f64 = maxpro_sum(design);
    let n_pairs: f64 = n as f64 * (n as f64 - 1.0) / 2.0;
    (inverse_product_sum / n_pairs).powf(1.0 / d as f64)
}

/// Using many iterations, choose the best LHD according to the MaxPro metric.
///
/// Arguments:
///     n_samples (u64): Number of samples the latin hypercube should contain
///     n_dim (u64): Number of dimensions the latin hypercube is composed of
///     n_iterations (u64): Number of iterations to generate latin hypercubes
///         for in order to determine an optimal design.
///
/// Returns:
///     Vec<Vec<f64>>: A optimized latin hypercube design that maximizes the MaxPro criterion
///
/// Panics:
///     n_samples == 0: n_samples must be positive and nonzero
///     n_dim == 0: n_dim must be positive and nonzero
///     n_iterations == 0: n_iterations must be positive and nonzero
pub fn build_maxpro_lhd(n_samples: u64, n_dim: u64, n_iterations: u64) -> Vec<Vec<f64>> {
    assert!(n_samples > 0, "n_samples must be positive and nonzero");
    assert!(n_dim > 0, "n_dim must be positive and nonzero");
    assert!(n_iterations > 0, "n_iterations must be positive and nonzero");
    let best_lhd_metric_pair: (Vec<Vec<f64>>, f64) = (0..n_iterations)
        .into_par_iter()
        .map(|_| {
            // Generate lhd, metric pairs in parallel via rayon's into_par_iter
            let lhd: Vec<Vec<f64>> = generate_lhd(n_samples, n_dim);
            let metric = maxpro_sum(&lhd);
            (lhd, metric)
        })
        .reduce(
            || (Vec::new(), f64::INFINITY), // Starting value
            // Iterate through at the op stage to find the lowest of any two pairs of comparison
            |(lhd1, metric1), (lhd2, metric2)| {
                if metric1 < metric2 {
                    (lhd1, metric1)
                } else {
                    (lhd2, metric2)
                }
            },
        );
    best_lhd_metric_pair.0 // only return the lhd, not the metric
}

#[cfg(feature = "pyo3-bindings")]
/// Calculate the maximum projection (maxpro) criterion for an input design
///
/// Args:
///     design (list[list[float]]): Design of interest
///
/// Returns:
///     float: Maximum projection criterion value
#[pyfunction(name = "maxpro_criterion")]
pub fn py_maxpro_criterion(design: Vec<Vec<f64>>) -> PyResult<f64> {
    if design.is_empty() {
        return Err(PyValueError::new_err("Design cannot be empty"));
    }
    Ok(maxpro_criterion(&design))
}

#[cfg(feature = "pyo3-bindings")]
/// Build a maxpro LHD
///
/// Args:
///     n_samples (int): Number of samples for the LHD
///     n_dim (int): Number of dimensions in which to generate points
///     n_iterations (int): Number of iterations to use to search for an optimal LHD
///
/// Returns:
///     list[list[float]]: A semi-optimal maxpro latin hypercube design
#[pyfunction(name = "build_maxpro_lhd")]
pub fn py_build_maxpro_lhd(
    n_samples: u64,
    n_dim: u64,
    n_iterations: u64,
) -> PyResult<Vec<Vec<f64>>> {
    if n_samples == 0 {
        return Err(PyValueError::new_err(
            "n_samples must be positive and nonzero",
        ));
    }
    if n_dim == 0 {
        return Err(PyValueError::new_err("n_dim must be positive and nonzero"));
    }
    if n_iterations == 0 {
        return Err(PyValueError::new_err(
            "n_iterations must be positive and nonzero",
        ));
    }
    if n_samples > usize::MAX as u64 {
        return Err(PyValueError::new_err("n_samples too large to index"));
    }
    if n_dim > usize::MAX as u64 {
        return Err(PyValueError::new_err("n_dim too large to index"));
    }
    Ok(build_maxpro_lhd(n_samples, n_dim, n_iterations))
}

#[test]
/// Ensures maxpro criterion is well-conditioned over randomized inputs
fn test_maxpro_criterion() {
    let n_iterations: u64 = 1000;
    let n_samples: u64 = 100;
    let n_dim: u64 = 5;
    for _i in 0..n_iterations {
        let lhd = generate_lhd(n_samples, n_dim);
        let maxpro_metric: f64 = maxpro_criterion(&lhd);
        assert!(maxpro_metric >= 0.0);
        assert!(maxpro_metric < f64::INFINITY)
    }
}

#[test]
/// Test that the value matches expectations to high tolerance
fn test_maxpro_criterion_value_1() {
    let design = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    let maxpro_value = maxpro_criterion(&design);
    let expected_value = 1.0;
    assert!((maxpro_value - expected_value).abs() < 10.0_f64.powf(-9.0))
}

#[test]
/// Test that the value matches expectations to high tolerance
fn test_maxpro_criterion_value_2() {
    let design = vec![vec![0.0, 2.0], vec![1.0, 0.0]];
    let maxpro_value = maxpro_criterion(&design);
    let expected_value = 0.5;
    assert!((maxpro_value - expected_value).abs() < 10.0_f64.powf(-9.0))
}
