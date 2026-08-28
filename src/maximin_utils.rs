#[cfg(test)]
use crate::lhd::generate_lhd;
#[cfg(test)]
use hegel::TestCase;
#[cfg(test)]
use hegel::generators as gs;
#[cfg(feature = "pyo3-bindings")]
use pyo3::PyResult;
#[cfg(feature = "pyo3-bindings")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;
#[cfg(test)]
use rand::SeedableRng;
#[cfg(test)]
use rand::rngs::StdRng;

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
pub fn calculate_l2_distance(point_a: &Vec<f64>, point_b: &Vec<f64>) -> f64 {
    assert_eq!(
        point_a.len(),
        point_b.len(),
        "point_a and point_b must have the same length"
    );
    assert!(!point_a.is_empty(), "Points must have nonzero length");
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

#[cfg(test)]
#[hegel::test(test_cases = 1000)]
/// Test that the maximin criterion obeys simple properties across many iterations
fn test_maximin_criterion(tc: TestCase) {
    let n_samples: u64 = tc.draw(gs::integers::<u64>().min_value(2).max_value(100));
    let n_dim: u64 = tc.draw(gs::integers::<u64>().min_value(1).max_value(10));
    let seed: u64 = 12345;

    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

    let lhd = generate_lhd(n_samples, n_dim, &mut rng);
    let maximin_metric: f64 = maximin_criterion(&lhd);
    assert!(maximin_metric >= 0.0);
    assert!(maximin_metric < f64::INFINITY)
}

#[cfg(test)]
#[hegel::test(test_cases = 100)]
/// Test that the maximin criterion is well-defined for a generated n×n design
/// whose entries are all bounded in [0, 1]
fn test_maximin_criterion_square_uniform(tc: TestCase) {
    let n: u64 = tc.draw(gs::integers::<u64>().min_value(2).max_value(50));
    let n_index: usize = n as usize;

    let row = gs::vecs(gs::floats::<f64>().min_value(0.0).max_value(1.0))
        .min_size(n_index)
        .max_size(n_index);
    let design: Vec<Vec<f64>> = tc.draw(gs::vecs(row).min_size(n_index).max_size(n_index));

    let maximin_metric: f64 = maximin_criterion(&design);
    assert!(maximin_metric >= 0.0);
    assert!(maximin_metric < f64::INFINITY);
}

#[cfg(test)]
#[hegel::test(test_cases = 100)]
/// Test that the L2 distance is symmetric and zero for identical points
fn test_calculate_l2_distance_symmetry(tc: TestCase) {
    let n_dim: u64 = tc.draw(gs::integers::<u64>().min_value(1).max_value(10));
    let n_index: usize = n_dim as usize;
    let tolerance: f64 = 10.0_f64.powf(-9.0);

    let point: Vec<f64> = tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(0.0).max_value(1.0))
            .min_size(n_index)
            .max_size(n_index),
    );
    let other: Vec<f64> = tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(0.0).max_value(1.0))
            .min_size(n_index)
            .max_size(n_index),
    );

    let distance_ab: f64 = calculate_l2_distance(&point, &other);
    let distance_ba: f64 = calculate_l2_distance(&other, &point);
    assert!((distance_ab - distance_ba).abs() < tolerance);

    let distance_aa: f64 = calculate_l2_distance(&point, &point);
    assert_eq!(distance_aa, 0.0);
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
