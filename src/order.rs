use core::f64;

use crate::maximin_utils::calculate_l2_distance;

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
