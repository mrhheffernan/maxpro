#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;
use rayon::prelude::*;
pub mod anneal;
pub mod enums;
pub mod lhd;
pub mod maximin_utils;
pub mod maxpro_utils;

pub fn build_lhd(
    n_samples: u64,
    n_dim: u64,
    n_iterations: u64,
    metric: Option<enums::Metrics>,
) -> Vec<Vec<f64>> {
    assert!(n_samples > 0, "n_samples must be positive and nonzero");
    assert!(n_dim > 0, "n_dim must be positive and nonzero");
    assert!(
        n_iterations > 0,
        "n_iterations must be positive and nonzero"
    );

    if metric.is_none() {
        let lhd: Vec<Vec<f64>> = lhd::generate_lhd(n_samples, n_dim);
        return lhd;
    }

    let metric_option = metric.unwrap();

    let (metric_fn, minimize) = match metric_option {
        enums::Metrics::MaxPro => (
            maxpro_utils::maxpro_criterion as fn(&Vec<Vec<f64>>) -> f64,
            true,
        ),
        enums::Metrics::MaxiMin => (
            maximin_utils::maximin_criterion as fn(&Vec<Vec<f64>>) -> f64,
            false,
        ),
    };

    let best_lhd_metric_pair: (Vec<Vec<f64>>, f64) = (0..n_iterations)
        .into_par_iter()
        .map(|_| {
            // Generate lhd, metric pairs in parallel via rayon's into_par_iter
            let lhd: Vec<Vec<f64>> = lhd::generate_lhd(n_samples, n_dim);
            let metric = metric_fn(&lhd);
            assert!(metric > 0.0, "Metric must be positive and nonzero");
            (lhd, metric)
        })
        .reduce(
            || (Vec::new(), f64::NEG_INFINITY), // Starting value
            // Iterate through at the op stage to find the highest of any two pairs of comparison
            |(lhd1, metric1), (lhd2, metric2)| {
                if minimize {
                    if metric1 < metric2 && metric1 > 0.0 {
                        (lhd1, metric1)
                    } else {
                        (lhd2, metric2)
                    }
                } else {
                    if metric1 > metric2 {
                        (lhd1, metric1)
                    } else {
                        (lhd2, metric2)
                    }
                }
            },
        );
    best_lhd_metric_pair.0 // only return the lhd, not the metric
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
