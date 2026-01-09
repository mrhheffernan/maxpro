use crate::lhd::generate_lhd;
#[cfg(feature = "pyo3-bindings")]
use pyo3::PyResult;
#[cfg(feature = "pyo3-bindings")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;
use rayon::prelude::*;
/// Calculate the L2 distance between two points, also known as the
/// Euclidean distance.
///
/// Arguments:
///     point_a (&Vec<f64>): First point
///     point_b (&Vec<f64>): Second point
///
/// Returns:
///     f64: Euclidean distance between them
///
/// Panics:
///     point_a.len() != point_b.len(): Points must have the same number of dimensions
///     point_a.len() == 0: Points cannot have 0 dimension
fn calculate_l2_distance(point_a: &Vec<f64>, point_b: &Vec<f64>) -> f64 {
    assert_eq!(
        point_a.len(),
        point_b.len(),
        "point_a and point_b must have the same length"
    );
    assert!(point_a.len() != 0, "Points must have nonzero length");
    // Iterator below is equivalent to this less idiomatic approach.
    // let mut distance: f64 = 0.0;
    // for i in 0..point_a.len() {
    //     distance += (point_a[i] - point_b[i]).powf(2.0)
    // }
    // distance.sqrt()
    point_a
        .iter()
        .zip(point_b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Calculate the minimum pairwise distance between points, to be maximized
///
/// Arguments:
///     design (&Vec<Vec<f64>>): Collection of input coordinates
///
/// Returns:
///     f64: Minimum pairwise distance between points
pub fn maximin_criterion(design: &Vec<Vec<f64>>) -> f64 {
    let n: usize = design.len();
    assert!(n > 0, "Cannot pass an empty design");
    let mut min_distance: f64 = f64::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let distance: f64 = calculate_l2_distance(&design[i], &design[j]);
            if distance < min_distance {
                min_distance = distance
            }
        }
    }
    min_distance
}

/// Using many iterations, select a LHD that maximizes the minimum pairwise distance between points.
///
/// Arguments:
///     n_samples (u64): Number of samples
///     n_dim (u64): Number of dimensions
///     n_iterations (u64): Number of iterations
///
/// Returns:
///     Vec<Vec<f64>>: Latin hypercube design that maximizes the minimum pairwise distance
///         between points from n_iterations of random sampling
pub fn build_maximin_lhd(n_samples: u64, n_dim: u64, n_iterations: u64) -> Vec<Vec<f64>> {
    assert!(n_samples > 0, "n_samples must be positive and nonzero");
    assert!(n_dim > 0, "n_dim must be positive and nonzero");
    assert!(
        n_iterations > 0,
        "n_iterations must be positive and nonzero"
    );
    let best_lhd_metric_pair: (Vec<Vec<f64>>, f64) = (0..n_iterations)
        .into_par_iter()
        .map(|_| {
            // Generate lhd, metric pairs in parallel via rayon's into_par_iter
            let lhd: Vec<Vec<f64>> = generate_lhd(n_samples, n_dim);
            let metric = maximin_criterion(&lhd);
            (lhd, metric)
        })
        .reduce(
            || (Vec::new(), f64::NEG_INFINITY), // Starting value
            // Iterate through at the op stage to find the highest of any two pairs of comparison
            |(lhd1, metric1), (lhd2, metric2)| {
                if metric1 > metric2 {
                    (lhd1, metric1)
                } else {
                    (lhd2, metric2)
                }
            },
        );
    best_lhd_metric_pair.0 // only return the lhd, not the metric
}

#[test]
/// Test that the maximin criterion obeys simple properties across many iterations
fn test_maximin_criterion() {
    let n_iterations: u64 = 1000;
    let n_samples: u64 = 100;
    let n_dim: u64 = 5;
    for _i in 0..n_iterations {
        let lhd = generate_lhd(n_samples, n_dim);
        let maximin_metric: f64 = maximin_criterion(&lhd);
        assert!(maximin_metric >= 0.0);
        assert!(maximin_metric < f64::INFINITY)
    }
}

#[test]
/// Test that the value matches expectations to high tolerance
fn test_maximin_criterion_value_1() {
    let design = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    let maximin_value = maximin_criterion(&design);
    let expected_value = 2.0_f64.powf(0.5);
    assert!((maximin_value - expected_value).abs() < 10.0_f64.powf(-9.0));
}

#[test]
/// Test that the value matches expectations to high tolerance
fn test_maximin_criterion_value_2() {
    let design = vec![vec![0.0, 2.0], vec![1.0, 0.0]];
    let maximin_value = maximin_criterion(&design);
    let expected_value = 5.0_f64.powf(0.5);
    assert!((maximin_value - expected_value).abs() < 10.0_f64.powf(-9.0))
}

#[cfg(feature = "pyo3-bindings")]
/// Calculate the maximin criterion
///
/// Args:
///     design (list[list[float]]): Input design
///
/// Returns:
///     float: Maximin criterion for the input design
#[pyfunction(name = "maximin_criterion")]
pub fn py_maximin_criterion(design: Vec<Vec<f64>>) -> PyResult<f64> {
    if design.is_empty() {
        return Err(PyValueError::new_err("Design cannot be empty"));
    }
    Ok(maximin_criterion(&design))
}

#[cfg(feature = "pyo3-bindings")]
/// Build a maximin LHD
///
/// Args:
///     n_samples (int): Number of samples for the LHD
///     n_dim (int): Number of dimensions in which to generate points
///     n_iterations (int): Number of iterations to use to search for an optimal LHD
///
/// Returns:
///     list[list[float]]: A semi-optimal maximin latin hypercube design
#[pyfunction(name = "build_maximin_lhd")]
pub fn py_build_maximin_lhd(
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
    Ok(build_maximin_lhd(n_samples, n_dim, n_iterations))
}
