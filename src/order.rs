#[cfg(test)]
use crate::lhd::generate_lhd;
use crate::maximin_utils::calculate_l2_distance;
#[cfg(any(test, feature = "pyo3-bindings"))]
use crate::maximin_utils::maximin_criterion;
#[cfg(any(test, feature = "pyo3-bindings"))]
use crate::maxpro_utils::maxpro_criterion;
use core::f64;
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

/// Order designs to select an optimal subset of the full design at each
/// stopping point. This can be used to produce preliminary results
/// with the best available subset's design characteristics.
///
/// This is presently only designed for unit hypercube designs.
///
/// Args:
///     lhd (Vec<Vec<f64>>): A design to order
///     metric (F): A metric function that maps &Vec<Vec<f64>> to f64
///     minimize (bool): Whether the metric is to be minimized
///
/// Returns:
///     Vec<Vec<f64>>: lhd with elements reordered for optimal run order
pub fn order_design<F>(lhd: Vec<Vec<f64>>, metric: F, minimize: bool) -> Vec<Vec<f64>>
where
    F: Fn(&Vec<Vec<f64>>) -> f64,
{
    if lhd.is_empty() {
        return lhd;
    }
    // First: Choose middle point
    let ndim: usize = lhd[0].len();
    if ndim == 0 {
        // Happens when vec![vec![]] is passed
        return lhd;
    }
    // Note: This assumes the design space is always a unit hypercube.
    // This is currently true, but must be updated if the condition is relaxed.
    let center: Vec<f64> = vec![0.5; ndim];
    let mut distance_to_center: f64 = f64::INFINITY;
    let mut center_point_index: usize = 0;

    for (i, row) in lhd.iter().enumerate() {
        let proposal_distance: f64 = calculate_l2_distance(&center, row);
        if proposal_distance < distance_to_center {
            distance_to_center = proposal_distance;
            center_point_index = i;
        }
    }

    let mut unordered_points: Vec<Vec<f64>> = lhd;

    let center_point: Vec<f64> = unordered_points.swap_remove(center_point_index);
    let mut ordered_design: Vec<Vec<f64>> = Vec::new();
    ordered_design.push(center_point);

    // Then for the remaining points, proceed in a loop
    // Attempt each point that has not been chosen
    // Find the one that produces the best metric
    // Select that point and update the design
    // Repeat
    while !unordered_points.is_empty() {
        let mut best_metric: f64 = match minimize {
            true => f64::INFINITY,
            false => f64::NEG_INFINITY,
        };
        let mut best_metric_index = 0;
        for (i, row_ref) in unordered_points.iter_mut().enumerate() {
            // Add the row under consideration, taking it from the original Vec
            let row = std::mem::take(row_ref);
            ordered_design.push(row);
            // Calculate the metric
            let metric_value = metric(&ordered_design);
            // Remove the candidate row
            let row = ordered_design.pop().unwrap();
            // Assign old value back to the original location in memory
            *row_ref = row;

            if minimize {
                if metric_value < best_metric {
                    best_metric = metric_value;
                    best_metric_index = i;
                }
            } else {
                if metric_value > best_metric {
                    best_metric = metric_value;
                    best_metric_index = i;
                }
            }
        }
        let next_best_row: Vec<f64> = unordered_points.swap_remove(best_metric_index);
        ordered_design.push(next_best_row);
    }
    ordered_design
}

#[test]
/// Test that the LHD has the same metric values before and after ordering
fn test_ordered_criteria_parity() {
    let n_iterations: u64 = 1000;
    let n_samples: u64 = 50;
    let n_dim: u64 = 5;
    let seed: u64 = 12345;
    let atol: f64 = 1e-10;

    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    for _i in 0..n_iterations {
        // Generate LHD
        let lhd = generate_lhd(n_samples, n_dim, &mut rng);
        let maxpro_metric_before: f64 = maxpro_criterion(&lhd);
        let maximin_metric_before: f64 = maximin_criterion(&lhd);

        // Order Design
        let ordered_design = order_design(lhd, maximin_criterion, false);

        // Ensure parity
        let maxpro_metric_after: f64 = maxpro_criterion(&ordered_design);
        let maximin_metric_after: f64 = maximin_criterion(&ordered_design);

        assert!(maxpro_metric_after >= 0.0);
        assert!(maxpro_metric_after < f64::INFINITY);
        assert!(maximin_metric_after >= 0.0);
        assert!(maximin_metric_after < f64::INFINITY);

        assert!((maximin_metric_before - maximin_metric_after).abs() < atol);
        assert!((maxpro_metric_before - maxpro_metric_after).abs() < atol);
    }
}

#[test]
/// Test that empty inner points are handled
fn test_empty_vecs() {
    let lhd = vec![vec![]];
    let ordered_lhd_minimize = order_design(lhd.clone(), maxpro_criterion, true);
    let ordered_lhd_maximize = order_design(lhd.clone(), maxpro_criterion, false);

    assert_eq!(lhd, ordered_lhd_minimize);
    assert_eq!(lhd, ordered_lhd_maximize);
}

#[cfg(feature = "pyo3-bindings")]
/// Order the design to optimize the run order for optimal subset ordering.
/// This is currently only for unit hypercube designs.
///
/// Args:
///     design (list[list[float]]): Design of interest
///     metric_name (str): Name of the metric of interest, one of ("maximin", "maxpro")
///
/// Returns:
///     list[list[float]]: Optimally-reordered design
#[pyfunction(name = "order_design")]
pub fn py_order_design(design: Vec<Vec<f64>>, metric_name: String) -> PyResult<Vec<Vec<f64>>> {
    if design.is_empty() {
        return Ok(design);
    }
    // Note: The `as` here is required for this to compile as the match arms having different
    // functions causes compilation errors despite matching signatures
    let (metric, minimize) = match metric_name.to_lowercase().as_str() {
        "maxpro" => (maxpro_criterion as fn(&Vec<Vec<f64>>) -> f64, true),
        "maximin" => (maximin_criterion as fn(&Vec<Vec<f64>>) -> f64, false),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown metric: '{}'. Available metrics are 'maxpro' and 'maximin'.",
                metric_name
            )));
        }
    };
    Ok(order_design(design, metric, minimize))
}
