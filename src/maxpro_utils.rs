use crate::lhd::generate_lhd;
#[cfg(feature = "pyo3-bindings")]
use pyo3::PyResult;
#[cfg(feature = "pyo3-bindings")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;

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
