use crate::lhd::generate_lhd;
use crate::maximin_utils::calculate_l2_distance;
use crate::maximin_utils::maximin_criterion;
use crate::maxpro_utils::maxpro_criterion;
use core::f64;
#[cfg(feature = "pyo3-bindings")]
use pyo3::PyResult;
#[cfg(feature = "pyo3-bindings")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;
use std::f32::EPSILON;

#[cfg(test)]
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Order designs to select an optimal subset of the full design at each
/// stopping point. This can be used to produce preliminary results
/// with the best available subset's design characteristics.
///
/// Args:
///     lhd (Vec<Vec<64>>): A design to order
///     metric (F): A metric function that maps &Vec<Vec<f64>> to f64
///     minimize (bool): Whether the metric is to be minimized
///
/// Returns:
///     Vec<Vec<f64>>: lhd with elements reordered for optimal run order
pub fn order_design<F>(lhd: Vec<Vec<f64>>, metric: F, minimize: bool) -> Vec<Vec<f64>>
where
    F: Fn(&Vec<Vec<f64>>) -> f64,
{
    // First: Choose middle point
    let ndim = lhd[0].len();
    let center = vec![0.5; ndim];
    let mut center_point = lhd[0].clone();
    let mut distance_to_center = calculate_l2_distance(&center, &center_point);
    let mut center_point_index = 0;

    for (i, row) in lhd.iter().enumerate() {
        let proposal_distance = calculate_l2_distance(&center, row);
        if proposal_distance < distance_to_center {
            center_point = row.clone();
            distance_to_center = proposal_distance;
            center_point_index = i;
        }
    }

    let mut ordered_design = vec![center_point];
    let mut unordered_points = lhd.clone();
    let _ = unordered_points.swap_remove(center_point_index);

    // Then for the remaining points, proceed in a loop
    // Attempt each point that has not been chosen
    // Find the one that produces the best metric
    // Select that point and update the design
    // Repeat
    while !unordered_points.is_empty() {
        let mut best_metric = match minimize {
            true => f64::INFINITY,
            false => f64::NEG_INFINITY,
        };
        let mut best_metric_index = 0;
        for (i, row) in unordered_points.iter().enumerate() {
            // There has to be a better, more efficient way to do this
            let mut proposed_design = ordered_design.clone();
            proposed_design.append(&mut vec![row.clone()]);
            let metric_value = metric(&proposed_design);
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
        let next_best_row = unordered_points.swap_remove(best_metric_index);
        ordered_design.append(&mut vec![next_best_row.clone()]);
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
        let ordered_design = order_design(lhd, maximin_criterion, true);

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

#[cfg(feature = "pyo3-bindings")] // WORK IN PROGRESS
/// Order the design to optimize the run order for optimal subset ordering
///
/// Args:
///     design (list[list[float]]): Design of interest
///     metric_name (str):
///
/// Returns:
///     float: Maximum projection criterion value
#[pyfunction(name = "order_design")]
pub fn py_order_design(design: Vec<Vec<f64>>, metric_name: String) -> PyResult<Vec<Vec<f64>>> {
    if design.is_empty() {
        return Err(PyValueError::new_err("Design cannot be empty"));
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
