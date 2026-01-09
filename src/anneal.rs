#[cfg(feature = "pyo3-bindings")]
use crate::maximin_utils::maximin_criterion;
#[cfg(feature = "pyo3-bindings")]
use crate::maxpro_utils::maxpro_criterion;
#[cfg(feature = "pyo3-bindings")]
use pyo3::PyResult;
#[cfg(feature = "pyo3-bindings")]
use pyo3::exceptions::PyValueError;
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
) -> PyResult<Vec<Vec<f64>>> {
    let metric = match metric_name.to_lowercase().as_str() {
        "maxpro" => maxpro_criterion,
        "maximin" => maximin_criterion,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown metric: '{}'. Available metrics are 'maxpro' and 'maximin'.",
                metric_name
            )));
        }
    };
    Ok(anneal_lhd(
        &design,
        n_iterations,
        initial_temp,
        cooling_rate,
        metric,
        minimize,
    ))
}
