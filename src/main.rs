use ndarray::ArrayBase; // Used for the generic array type in the function signature.
use ndarray::{Array, Data, Ix2, s}; // Import 's!' for slicing.
use rand::Rng;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

fn generate_coords(n_samples: i32) -> Vec<(f64, f64)> {
    (0..n_samples)
        .into_par_iter()
        .map_init(rand::rng, |rng, _| {
            (rng.random_range(0.0..1.0), rng.random_range(0.0..1.0))
        })
        .collect()
}

fn generate_lhd(n_samples: usize, n_dim: usize) -> Vec<Vec<f64>> {
    // initialize empty lhd
    let mut rng = rand::rng();
    let mut lhd = vec![vec![0.0; n_dim]; n_samples];

    // For each dimension, start by generating a shuffle
    for j in 0..n_dim {
        let mut permutation: Vec<usize> = (0..n_samples).collect(); // might need a .collect()?
        // Shuffle the 0..n_samples iterator
        permutation.shuffle(&mut rng);

        for i in 0..n_samples {
            // okay, now for the tricky bit.
            // Get the range of the interval for this sample
            let interval_start = permutation[i] as f64 / n_samples as f64;
            let interval_end = (permutation[i] + 1) as f64 / n_samples as f64;
            // Generate a random sample in the box
            let sample = rng.random_range(interval_start..interval_end);
            lhd[i][j] = sample;
        }
    }
    lhd
}

// Accepts any array (S) where S can be borrowed to view (&) f64.
pub fn maxpro_criterion<S>(design: &ArrayBase<S, Ix2>) -> f64
where
    S: Data<Elem = f64>,
{
    // The storage type S must contain f64 elements
    /*
    Calculates the internal sum term of the MaxPro criterion (psi(D)).
    This sum is the term that is directly minimized in the optimization
    process.

    The minimized term is: sum_{i<j} [ 1 / product_{l=1}^{d} (x_il - x_jl)^2 ]

    Args:
        design: An ndarray (n x d) representing the design points,
                normalized to the unit hypercube [0, 1]^d.

    Returns:
        The value of the internal sum (the MaxPro Sum Metric).
    */
    let (n, _d) = design.dim();

    if n < 2 {
        return 0.0;
    }

    let mut inverse_product_sum = 0.0;
    let epsilon = 1e-12; // Small constant to prevent division by zero

    for i in 0..n {
        for j in (i + 1)..n {
            // Calculate the product term: product_{l=1}^{d} (x_il - x_jl)^2
            let row_i_slice = design.slice(s![i, ..]);
            let row_j_slice = design.slice(s![j, ..]);

            // // 2. Subtract: .to_owned() converts the second slice to an owned array (Array1)
            // //    to allow the subtraction operation.
            let diffs = row_i_slice.to_owned() - row_j_slice;

            let diffs_sq = diffs.map(|x| x * x);
            let product_of_squared_diffs: f64 = diffs_sq.product() + epsilon;

            // // Sum the inverse products
            inverse_product_sum += 1.0 / product_of_squared_diffs;
        }
    }

    inverse_product_sum
}

fn main() {
    let n_samples: i32 = 64;
    let coords = generate_coords(n_samples);

    let n = coords.len(); // Number of rows (3)
    let d = 2; // Number of columns (2)

    // 2. Convert to a flat vector using iteration methods
    let flat_data: Vec<f64> = coords
        .into_iter()
        // map each tuple (x, y) into a temporary array [x, y]
        .flat_map(|(x, y)| [x, y].into_iter())
        // collect the results into a single flat Vec<f64>
        .collect();
    let arr = Array::from_shape_vec((n, d), flat_data).unwrap();
    let maxpro = maxpro_criterion(&arr);
    println!("Maxpro: {maxpro}");

    let lhd = generate_lhd(n_samples as usize, 2);
    let lhd_array = Array::from(lhd);
    let maxpro2 = maxpro_criterion(lhd_array);
    println!("Maxpro 2: {maxpro2}");
}
