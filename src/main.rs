use ndarray::ArrayBase; // Used for the generic array type in the function signature.
use ndarray::{Array, Array2, Data, Ix2, ShapeError, s}; // Import 's!' for slicing.
use plotters::prelude::*;
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

fn plot_x_vs_y(data: Vec<Vec<f64>>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("xy_scatter_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    // 1. Prepare data and determine axis bounds
    // We transform the Vec<Vec<f64>> into a Vec<(f64, f64)> of (x, y) pairs.
    let points: Vec<(f64, f64)> = data
        .into_iter()
        .filter_map(|p| {
            // Ensure the point has at least 2 dimensions [x, y]
            if p.len() >= 2 {
                // Return (x, y) tuple
                Some((p[0], p[1]))
            } else {
                None
            }
        })
        .collect();

    // Find the min/max X and Y for axis scaling
    let (min_x, max_x) = points
        .iter()
        .map(|(x, _)| *x)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_x, max_x), x| {
            (min_x.min(x), max_x.max(x))
        });
    let (min_y, max_y) = points
        .iter()
        .map(|(_, y)| *y)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), y| {
            (min_y.min(y), max_y.max(y))
        });

    // Add a small buffer and use the EXCLUSIVE range operator (..)
    let x_range = (min_x - 0.1)..(max_x + 0.1); 
    let y_range = (min_y - 0.1)..(max_y + 0.1); 

    // 2. Create the chart context
    let mut chart = ChartBuilder::on(&root)
        // ... other settings ...
        // This call now satisfies the Ranged trait bound:
        .build_cartesian_2d(x_range, y_range)?;

    chart
        .configure_mesh()
        .x_desc("X Coordinate")
        .y_desc("Y Coordinate")
        .draw()?;

    // 3. Draw the single series of points
    chart
        .draw_series(
            // The PointSeries creation, without .mark_as_owned()
            PointSeries::<_, _, Circle<_, _>, _>::new(
                points,        // Data is Vec<(f64, f64)>
                5,             // Radius
                BLUE.filled(), // Color
            ),
        )?
        .label("Design Points")
        .legend(move |(x, y)| Circle::new((x, y), 5, BLUE.filled()));

    root.present()?;
    Ok(())
}

fn build_maxpro_lhd(n_samples: i32, n_iterations: i32, n_dim: usize, plot: bool) -> Vec<Vec<f64>> {
    let mut best_metric = f64::INFINITY;
    let mut best_lhd = vec![vec![0.0; n_dim]; n_samples];
    for _i in 0..n_iterations {
        let lhd = generate_lhd(n_samples as usize, n_dim);
        let lhd_array = convert_design_to_array2(lhd.clone()).unwrap();
        let maxpro_metric = maxpro_criterion(&lhd_array);
        if maxpro_metric < best_metric {
            best_lhd = lhd.clone();
            best_metric = maxpro_metric;
            if plot {
                let _ = plot_x_vs_y(best_lhd.clone());
            }
            println!("Best metric: {best_metric}")
        }
    }
    best_lhd
}

fn main() {
    let n_samples: i32 = 100;
    const N_ITERATIONS: i32 = 10000;
    let plot: bool = false;
    let maxpro_lhd = build_maxpro_lhd(n_samples, N_ITERATIONS, 2, plot);
}
