use crate::maximin_utils::calculate_l2_distance;

/// Order designs to select an optimal subset of the full design at each
/// stopping point. This can be used to produce preliminary results
/// with the best available subset's design characteristics.
pub fn order_design<F>(lhd: Vec<Vec<f64>>, metric: F, minimize: bool, seed: u64) -> Vec<Vec<f64>>
where
    F: Fn(&Vec<Vec<f64>>) -> f64,
{
    // First: Choose middle point
    let ndim = lhd[0].len();
    let n_samples = lhd.len();
    let center = vec![0.5; ndim];
    let mut center_point = lhd[0].clone();
    let mut distance_to_center = calculate_l2_distance(&center, &center_point);

    for i in 0..n_samples {
        let proposal_distance = calculate_l2_distance(&center, &lhd[i]);
        if proposal_distance < distance_to_center {
            center_point = lhd[i].clone();
            distance_to_center = proposal_distance;
        }
    }

    println!("Point closest to center is {} away", distance_to_center);

    let mut ordered_design = vec![center_point];

    // Then for the remaining points, proceed in a loop
    // Attempt each point that has not been chosen
    // Find the one that produces the best metric
    // Select that point and update the design
    // Repeat
    ordered_design
}
