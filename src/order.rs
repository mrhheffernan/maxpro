use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Order designs to select an optimal subset of the full design at each 
/// stopping point. This can be used to produce preliminary results
/// with the best available subset's design characteristics.
pub fn order_design(lhd: Vec<Vec<f64>>, metric: F, minimize: bool, seed: u64) -> Vec<Vec<f64>> {
    // First: Choose middle point

    // Then for the remaining points, proceed in a loop
    // Attempt each point that has not been chosen
    // Find the one that produces the best metric
    // Select that point and update the design
    // Repeat
}