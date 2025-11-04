#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;

pub mod utils {
    use plotters::prelude::*;
    #[cfg(feature = "pyo3-bindings")]
    use pyo3::prelude::*;
    use rand::Rng;
    use rand::prelude::{SliceRandom, ThreadRng};

    pub fn plot_x_vs_y(
        data: &Vec<Vec<f64>>,
        output_path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if data.is_empty() || data[0].len() < 2 {
            return Err(From::from(
                "Data for plot_x_vs_y must have at least 2 columns and 1 sample.",
            ));
        }

        let root = BitMapBackend::new(output_path, (640, 480)).into_drawing_area();
        root.fill(&WHITE)?;

        // 1. Prepare data and determine axis bounds
        let points: Vec<(f64, f64)> = data.iter().map(|row| (row[0], row[1])).collect();

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
            .draw_series(PointSeries::<_, _, Circle<_, _>, _>::new(
                points,        // Data is Vec<(f64, f64)>
                5,             // Radius
                BLUE.filled(), // Color
            ))?
            .label("Design Points")
            .legend(move |(x, y)| Circle::new((x, y), 5, BLUE.filled()));

        root.present()?;
        Ok(())
    }

    pub fn generate_lhd(n_samples: usize, n_dim: usize) -> Vec<Vec<f64>> {
        // initialize empty lhd
        // TODO: Make this seedable
        let mut rng: ThreadRng = rand::rng();
        let mut lhd: Vec<Vec<f64>> = vec![vec![0.0; n_dim]; n_samples];

        // For each dimension, start by generating a shuffle
        for j in 0..n_dim {
            let mut permutation: Vec<usize> = (0..n_samples).collect();
            // Shuffle the 0..n_samples iterator
            permutation.shuffle(&mut rng);

            for i in 0..n_samples {
                // Get the range of the interval for this sample
                let interval_start = permutation[i] as f64 / n_samples as f64;
                let interval_end = (permutation[i] + 1) as f64 / n_samples as f64;
                // Generate a random sample in the box
                let sample: f64 = rng.random_range(interval_start..interval_end);
                // lhd[[i, j]] = sample;
                lhd[i][j] = sample;
            }
        }
        lhd
    }

    pub fn maxpro_criterion(design: &Vec<Vec<f64>>) -> f64 {
        /*
        Calculates the internal sum term of the MaxPro criterion (psi(D)).
        This sum is the term that is directly minimized in the optimization
        process.

        The minimized term is: sum_{i<j} [ 1 / product_{l=1}^{d} (x_il - x_jl)^2 ]

        Args:
            design: An Vec<Vec<f64>> (n x d) representing the design points,
                    normalized to the unit hypercube [0, 1]^d.

        Returns:
            The value of the internal sum (the MaxPro Sum Metric).
        */

        let n: usize = design.len();
        let d: usize = design[0].len();

        if n < 2 {
            return 0.0;
        }

        let mut inverse_product_sum: f64 = 0.0;
        let epsilon: f64 = 1e-12; // Small constant to prevent division by zero
        let n_pairs: f64 = n as f64 * (n as f64 - 1.0) / 2.0;

        for i in 0..n {
            for j in (i + 1)..n {
                // Calculate the product term: product_{l=1}^{d} (x_il - x_jl)^2
                let row_i: &Vec<f64> = &design[i];
                let row_j: &Vec<f64> = &design[j];
                let mut product_of_squared_diffs: f64 = 1.0;

                // Use zip to iterate over both rows simultaneously
                for (x_i_l, x_j_l) in row_i.iter().zip(row_j.iter()) {
                    let diff: f64 = x_i_l - x_j_l;
                    let diff_sq: f64 = diff * diff;
                    product_of_squared_diffs *= diff_sq;
                }

                // Add epsilon to prevent division by zero
                product_of_squared_diffs += epsilon;

                // // Sum the inverse products
                inverse_product_sum += 1.0 / product_of_squared_diffs;
            }
        }

        (inverse_product_sum / n_pairs).powf(1.0 / d as f64)
    }

    pub fn build_maxpro_lhd(n_samples: usize, n_dim: usize, n_iterations: usize) -> Vec<Vec<f64>> {
        let mut best_metric = f64::INFINITY;
        let mut best_lhd: Vec<Vec<f64>> = vec![vec![0.0; n_dim]; n_samples as usize];
        for _i in 0..n_iterations {
            let lhd: Vec<Vec<f64>> = generate_lhd(n_samples, n_dim);
            let maxpro_metric: f64 = maxpro_criterion(&lhd);
            if maxpro_metric < best_metric {
                best_lhd = lhd.clone();
                best_metric = maxpro_metric;
            }
        }
        best_lhd
    }

    #[cfg(feature = "pyo3-bindings")]
    #[pyfunction(name = "generate_lhd")]
    pub fn py_generate_lhd(n_samples: usize, n_dim: usize) -> Vec<Vec<f64>> {
        generate_lhd(n_samples, n_dim)
    }

    #[cfg(feature = "pyo3-bindings")]
    #[pyfunction(name = "maxpro_criterion")]
    pub fn py_maxpro_criterion(design: Vec<Vec<f64>>) -> f64 {
        maxpro_criterion(&design)
    }

    #[cfg(feature = "pyo3-bindings")]
    #[pyfunction(name = "build_maxpro_lhd")]
    pub fn py_build_maxpro_lhd(
        n_samples: usize,
        n_iterations: usize,
        n_dim: usize,
    ) -> Vec<Vec<f64>> {
        build_maxpro_lhd(n_samples, n_iterations, n_dim)
    }
}

// Python module definition
#[cfg(feature = "pyo3-bindings")]
#[pymodule]
fn maxpro(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(utils::py_build_maxpro_lhd, m)?)?;
    m.add_function(wrap_pyfunction!(utils::py_generate_lhd, m)?)?;
    m.add_function(wrap_pyfunction!(utils::py_maxpro_criterion, m)?)?;
    Ok(())
}
