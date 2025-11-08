#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;

pub mod lhd {
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
                let interval_start: f64 = permutation[i] as f64 / n_samples as f64;
                let interval_end: f64 = (permutation[i] + 1) as f64 / n_samples as f64;
                // Generate a random sample in the box
                let sample: f64 = rng.random_range(interval_start..interval_end);
                // lhd[[i, j]] = sample;
                lhd[i][j] = sample;
            }
        }
        lhd
    }

    #[test]
    fn test_generate_lhd() {
        /*
        Makes a simple test of a latin hypercube checking that,
        for any interval in any dimension, there should be only one sample.
        */
        let n_samples: usize = 100;
        let n_dim: usize = 4;
        let design = generate_lhd(n_samples, n_dim);

        for dimension in 0..n_dim {
            // n_samples in an LHD is the same as the number of intervals to fill
            let mut counts = vec![0; n_samples];
            for interval in 0..n_samples {
                // back-engineer the interval
                let sample = design[interval][dimension];
                let sample_interval_idx = (sample * n_samples as f64).floor() as usize;
                counts[sample_interval_idx] += 1;
            }
            assert!(counts.iter().all(|&c| c == 1))
        }
    }

    #[cfg(feature = "pyo3-bindings")]
    #[pyfunction(name = "generate_lhd")]
    pub fn py_generate_lhd(n_samples: usize, n_dim: usize) -> Vec<Vec<f64>> {
        generate_lhd(n_samples, n_dim)
    }
}

pub mod maxpro_utils {
    use crate::lhd::generate_lhd;
    #[cfg(feature = "pyo3-bindings")]
    use pyo3::prelude::*;
    fn maxpro_sum(design: &Vec<Vec<f64>>) -> f64 {
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
        if n < 2 {
            return 0.0;
        }

        let mut inverse_product_sum: f64 = 0.0;
        let epsilon: f64 = 1e-12; // Small constant to prevent division by zero

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

        inverse_product_sum
    }

    pub fn maxpro_criterion(design: &Vec<Vec<f64>>) -> f64 {
        /* Calculates the full, complete MaxPro criterion */
        let n: usize = design.len();
        if n < 2 {
            return 0.0;
        }
        let d: usize = design[0].len();
        let inverse_product_sum: f64 = maxpro_sum(design);
        let n_pairs: f64 = n as f64 * (n as f64 - 1.0) / 2.0;
        (inverse_product_sum / n_pairs).powf(1.0 / d as f64)
    }

    pub fn build_maxpro_lhd(n_samples: usize, n_dim: usize, n_iterations: usize) -> Vec<Vec<f64>> {
        let mut best_metric = f64::INFINITY;
        let mut best_lhd: Vec<Vec<f64>> = vec![vec![0.0; n_dim]; n_samples];
        for _i in 0..n_iterations {
            let lhd: Vec<Vec<f64>> = generate_lhd(n_samples, n_dim);
            let maxpro_metric: f64 = maxpro_sum(&lhd);
            if maxpro_metric < best_metric {
                best_lhd = lhd.clone();
                best_metric = maxpro_metric;
            }
        }
        best_lhd
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

    #[test]
    fn test_maxpro_criterion() {
        let n_iterations: usize = 10;
        let n_samples: usize = 100;
        let n_dim: usize = 5;
        for _i in 0..n_iterations {
            let lhd = generate_lhd(n_samples, n_dim);
            let maxpro_metric: f64 = maxpro_criterion(&lhd);
            assert!(maxpro_metric >= 0.0);
            assert!(maxpro_metric < f64::INFINITY)
        }
    }
}

pub mod maximin_utils {
    use crate::lhd::generate_lhd;
    #[cfg(feature = "pyo3-bindings")]
    use pyo3::prelude::*;
    fn calculate_l2_distance(point_a: &Vec<f64>, point_b: &Vec<f64>) -> f64 {
        assert!(point_a.len() == point_b.len());
        let mut distance: f64 = 0.0;
        for i in 0..point_a.len() {
            distance += (point_a[i] - point_b[i]).powf(2.0)
        }
        distance.sqrt()
    }

    fn maximin_criterion(design: &Vec<Vec<f64>>) -> f64 {
        let n: usize = design.len();
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

    pub fn build_maximin_lhd(n_samples: usize, n_dim: usize, n_iterations: usize) -> Vec<Vec<f64>> {
        let mut best_metric = 0.0;
        let mut best_lhd: Vec<Vec<f64>> = vec![vec![0.0; n_dim]; n_samples];
        for _i in 0..n_iterations {
            let lhd: Vec<Vec<f64>> = generate_lhd(n_samples, n_dim);
            let metric: f64 = maximin_criterion(&lhd);
            if metric > best_metric {
                best_lhd = lhd.clone();
                best_metric = metric;
            }
        }
        best_lhd
    }

    #[cfg(feature = "pyo3-bindings")]
    #[pyfunction(name = "maximincriterion")]
    pub fn py_maximin_criterion(design: Vec<Vec<f64>>) -> f64 {
        maximin_criterion(&design)
    }

    #[cfg(feature = "pyo3-bindings")]
    #[pyfunction(name = "build_maximin_lhd")]
    pub fn py_build_maximin_lhd(
        n_samples: usize,
        n_iterations: usize,
        n_dim: usize,
    ) -> Vec<Vec<f64>> {
        build_maximin_lhd(n_samples, n_iterations, n_dim)
    }
}

// Python module definition
#[cfg(feature = "pyo3-bindings")]
#[pymodule]
fn maxpro(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lhd::py_generate_lhd, m)?)?;
    m.add_function(wrap_pyfunction!(maxpro_utils::py_build_maxpro_lhd, m)?)?;
    m.add_function(wrap_pyfunction!(maxpro_utils::py_maxpro_criterion, m)?)?;
    m.add_function(wrap_pyfunction!(maximin_utils::py_build_maximin_lhd, m)?)?;
    m.add_function(wrap_pyfunction!(maximin_utils::py_maximin_criterion, m)?)?;
    Ok(())
}
