#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;
pub mod lhd;
pub mod maxpro_utils;

pub mod maximin_utils {
    use crate::lhd::generate_lhd;
    #[cfg(feature = "pyo3-bindings")]
    use pyo3::prelude::*;
    use rayon::prelude::*;
    /// Calculate the L2 distance between two points, also known as the
    /// Euclidean distance.
    ///
    /// Arguments:
    ///     point_a (&Vec<f64>): First point
    ///     point_b (&Vec<f64>): Second point
    ///
    /// Returns:
    ///     f64: Euclidean distance between them
    ///
    /// Panics:
    ///     point_a.len() != point_b.len(): Points must have the same number of dimensions
    ///     point_a.len() == 0: Points cannot have 0 dimension
    fn calculate_l2_distance(point_a: &Vec<f64>, point_b: &Vec<f64>) -> f64 {
        assert_eq!(
            point_a.len(),
            point_b.len(),
            "point_a and point_b must have the same length"
        );
        assert!(point_a.len() != 0, "Points must have nonzero length");
        // Iterator below is equivalent to this less idiomatic approach.
        // let mut distance: f64 = 0.0;
        // for i in 0..point_a.len() {
        //     distance += (point_a[i] - point_b[i]).powf(2.0)
        // }
        // distance.sqrt()
        point_a
            .iter()
            .zip(point_b)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate the minimum pairwise distance between points, to be maximized
    ///
    /// Arguments:
    ///     design (&Vec<Vec<f64>>): Collection of input coordinates
    ///
    /// Returns:
    ///     f64: Minimum pairwise distance between points
    pub fn maximin_criterion(design: &Vec<Vec<f64>>) -> f64 {
        let n: usize = design.len();
        assert!(n > 0, "Cannot pass an empty design");
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

    /// Using many iterations, select a LHD that maximizes the minimum pairwise distance between points.
    ///
    /// Arguments:
    ///     n_samples (u64): Number of samples
    ///     n_dim (u64): Number of dimensions
    ///     n_iterations (u64): Number of iterations
    ///
    /// Returns:
    ///     Vec<Vec<f64>>: Latin hypercube design that maximizes the minimum pairwise distance
    ///         between points from n_iterations of random sampling
    pub fn build_maximin_lhd(n_samples: u64, n_dim: u64, n_iterations: u64) -> Vec<Vec<f64>> {
        assert!(n_samples > 0, "n_samples must be positive and nonzero");
        assert!(n_dim > 0, "n_dim must be positive and nonzero");
        assert!(
            n_iterations > 0,
            "n_iterations must be positive and nonzero"
        );
        let best_lhd_metric_pair: (Vec<Vec<f64>>, f64) = (0..n_iterations)
            .into_par_iter()
            .map(|_| {
                // Generate lhd, metric pairs in parallel via rayon's into_par_iter
                let lhd: Vec<Vec<f64>> = generate_lhd(n_samples, n_dim);
                let metric = maximin_criterion(&lhd);
                (lhd, metric)
            })
            .reduce(
                || (Vec::new(), f64::NEG_INFINITY), // Starting value
                // Iterate through at the op stage to find the highest of any two pairs of comparison
                |(lhd1, metric1), (lhd2, metric2)| {
                    if metric1 > metric2 {
                        (lhd1, metric1)
                    } else {
                        (lhd2, metric2)
                    }
                },
            );
        best_lhd_metric_pair.0 // only return the lhd, not the metric
    }

    #[test]
    /// Test that the maximin criterion obeys simple properties across many iterations
    fn test_maximin_criterion() {
        let n_iterations: u64 = 1000;
        let n_samples: u64 = 100;
        let n_dim: u64 = 5;
        for _i in 0..n_iterations {
            let lhd = generate_lhd(n_samples, n_dim);
            let maximin_metric: f64 = maximin_criterion(&lhd);
            assert!(maximin_metric >= 0.0);
            assert!(maximin_metric < f64::INFINITY)
        }
    }

    #[test]
    /// Test that the value matches expectations to high tolerance
    fn test_maximin_criterion_value_1() {
        let design = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let maximin_value = maximin_criterion(&design);
        let expected_value = 2.0_f64.powf(0.5);
        assert!((maximin_value - expected_value).abs() < 10.0_f64.powf(-9.0));
    }

    #[test]
    /// Test that the value matches expectations to high tolerance
    fn test_maximin_criterion_value_2() {
        let design = vec![vec![0.0, 2.0], vec![1.0, 0.0]];
        let maximin_value = maximin_criterion(&design);
        let expected_value = 5.0_f64.powf(0.5);
        assert!((maximin_value - expected_value).abs() < 10.0_f64.powf(-9.0))
    }

    #[cfg(feature = "pyo3-bindings")]
    /// Calculate the maximin criterion
    ///
    /// Args:
    ///     design (list[list[float]]): Input design
    ///
    /// Returns:
    ///     float: Maximin criterion for the input design
    #[pyfunction(name = "maximin_criterion")]
    pub fn py_maximin_criterion(design: Vec<Vec<f64>>) -> f64 {
        maximin_criterion(&design)
    }

    #[cfg(feature = "pyo3-bindings")]
    /// Build a maximin LHD
    ///
    /// Args:
    ///     n_samples (int): Number of samples for the LHD
    ///     n_dim (int): Number of dimensions in which to generate points
    ///     n_iterations (int): Number of iterations to use to search for an optimal LHD
    ///
    /// Returns:
    ///     list[list[float]]: A semi-optimal maximin latin hypercube design
    #[pyfunction(name = "build_maximin_lhd")]
    pub fn py_build_maximin_lhd(n_samples: u64, n_dim: u64, n_iterations: u64) -> Vec<Vec<f64>> {
        build_maximin_lhd(n_samples, n_dim, n_iterations)
    }
}

pub mod anneal {
    #[cfg(feature = "pyo3-bindings")]
    use crate::maximin_utils::maximin_criterion;
    #[cfg(feature = "pyo3-bindings")]
    use crate::maxpro_utils::maxpro_criterion;
    #[cfg(feature = "pyo3-bindings")]
    use pyo3::prelude::*;
    use rand::Rng;
    use rand::prelude::ThreadRng;

    /// Simulated annealing for improving (maximizing or minimizing) a given metric.
    ///
    /// Arguments:
    ///     design (&Vec<Vec<f64>>): Initial design of points
    ///     n_iterations (u64): Number of iterations for annealing
    ///     initial_temp (f64): Initial temperature, used to control the metropolis algorithm acceptance
    ///     cooling rate (f64): Cooling rate, used to simulate annealing by reducing the metropolis algorithm acceptance
    ///     metric (F): A callable function that maps &Vec<Vec<f64>> to f64, used to minimize or maximize
    ///     minimize (bool): Whether to minimize or maximize the metric
    ///
    /// Returns:
    ///     Vec<Vec<f64>>: A metric-optimized collection of points, not necessarily a latin hypercube.
    pub fn anneal_lhd<F>(
        design: &Vec<Vec<f64>>,
        n_iterations: u64,
        initial_temp: f64,
        cooling_rate: f64,
        metric: F,
        minimize: bool,
    ) -> Vec<Vec<f64>>
    where
        F: Fn(&Vec<Vec<f64>>) -> f64,
    {
        if design.is_empty() {
            return design.to_vec();
        }
        assert!(
            n_iterations > 0,
            "n_iterations must be positive and nonzero"
        );
        assert!(
            initial_temp > 0.0,
            "initial_temp must be positive and nonzero"
        );
        // Set max step size as +/- 1% of the size of the design interval
        let n_samples: usize = design.len();
        let n_dim: usize = design[0].len();
        let step_size: f64 = 0.01 / n_samples as f64;
        let mut temp = initial_temp;
        // TODO: Make this seedable
        let mut rng: ThreadRng = rand::rng();

        let mut best_design = design.clone();
        let mut best_metric = metric(design);

        // Retain global best so annealing can never make a result worse
        let mut global_best_design = design.clone();
        let mut global_best_metric = best_metric;

        for _it in 0..n_iterations {
            // Modify the design
            let mut annealed_design = best_design.clone();
            for i in 0..n_samples {
                for j in 0..n_dim {
                    // Perturb the point, ensuring the point remains on the unit interval.
                    annealed_design[i][j] = (annealed_design[i][j]
                        + rng.random_range(-step_size..step_size))
                    .clamp(0.0, 1.0)
                }
            }

            // Calculate new metric
            let new_metric = metric(&annealed_design);
            let mut metric_diff = new_metric - best_metric;
            if !minimize {
                // Invert for maximization
                metric_diff *= -1.0
            }

            // Metropolis probability of acceptance
            let p_acceptance: f64 = if metric_diff < 0.0 {
                1.0
            } else {
                (-metric_diff / temp).exp()
            };

            // Metropolis acceptance
            if rng.random_range(0.0..1.0) < p_acceptance {
                best_design = annealed_design;
                best_metric = new_metric;
            }

            // Ensure the global best is returned
            if (minimize && best_metric < global_best_metric)
                || (!minimize && best_metric > global_best_metric)
            {
                global_best_design = best_design.clone();
                global_best_metric = best_metric;
            }

            // Cool for the next iteration
            temp *= cooling_rate;
        }

        global_best_design
    }

    #[cfg(feature = "pyo3-bindings")]
    /// Anneal a latin hypercube to optimize its metric value.
    /// Available metrics are "maxpro" and "maximin".
    ///
    /// Args:
    ///     design (list[list[float]]): Input latin hypercube design
    ///     n_iterations (int): Number of optimizations to search for a better candidate
    ///     initial_temp (float): Initial temperature for annealing
    ///     cooling_rate (float): Cooling rate for annealing
    ///     metric_name (str): metric name; options are "maxpro" and "maximin"
    ///     minimize (bool): Whether to minimize the metric
    ///
    /// Returns:
    ///     list[list[float]]: Optimized latin hypercube design
    #[pyfunction(name = "anneal_lhd")]
    pub fn py_anneal_lhd(
        design: Vec<Vec<f64>>,
        n_iterations: u64,
        initial_temp: f64,
        cooling_rate: f64,
        metric_name: String,
        minimize: bool,
    ) -> Vec<Vec<f64>> {
        let metric = match metric_name.to_lowercase().as_str() {
            "maxpro" => maxpro_criterion,
            "maximin" => maximin_criterion,
            _ => panic!(
                "Unknown metric: '{}'. Available metrics are 'maxpro' and 'maximin'.",
                metric_name
            ),
        };
        anneal_lhd(
            &design,
            n_iterations,
            initial_temp,
            cooling_rate,
            metric,
            minimize,
        )
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
    m.add_function(wrap_pyfunction!(anneal::py_anneal_lhd, m)?)?;
    Ok(())
}
