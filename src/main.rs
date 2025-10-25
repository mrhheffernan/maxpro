use ndarray::ArrayBase; // Used for the generic array type in the function signature.
use ndarray::{Array, Array2, Data, Ix2, ShapeError, s}; // Import 's!' for slicing.
use rand::Rng;
use rand::prelude::SliceRandom;

fn generate_lhd(n_samples: usize, n_dim: usize) -> Vec<Vec<f64>> {
    // initialize empty lhd
    // TODO: Make this seedable
    let mut rng = rand::rng();
    let mut lhd = vec![vec![0.0; n_dim]; n_samples];

    // For each dimension, start by generating a shuffle
    for j in 0..n_dim {
        let mut permutation: Vec<usize> = (0..n_samples).collect(); // might need a .collect()?
        // Shuffle the 0..n_samples iterator
        permutation.shuffle(&mut rng);

        for i in 0..n_samples {
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

fn convert_design_to_array2(design: Vec<Vec<f64>>) -> Result<Array2<f64>, ShapeError> {
    // Function to flatten the vector of vectors, slight tweaks from vec of tuples.
    let n = design.len();

    if n == 0 {
        return Array2::from_shape_vec((0, 0), Vec::new());
    }

    // Use the length of the first row for the column count (d)
    let d = design[0].len();

    // Flatten the Vec<Vec<f64>> into a single contiguous Vec<f64>
    let flat_data: Vec<f64> = design
        .into_iter()
        .flat_map(|inner_vec| inner_vec.into_iter())
        .collect();

    // Construct the Array2<f64> from the flat data and the shape (n, d)
    Array2::from_shape_vec((n, d), flat_data)
}

fn main() {
    let n_samples: i32 = 64;
    const N_ITERATIONS: i32 = 1000;
    let mut best_metric = 10e12;

    for _i in 0..N_ITERATIONS {
        let lhd = generate_lhd(n_samples as usize, 2);
        let lhd_array = convert_design_to_array2(lhd).unwrap();
        let maxpro2 = maxpro_criterion(&lhd_array);
        // println!("Maxpro 2: {maxpro2}");
        if maxpro2 < best_metric {
            let best_lhd = lhd_array;
            best_metric = maxpro2;
            println!("Best metric: {best_metric}")
        }
    }
}
