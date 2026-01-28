#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;
pub mod anneal;
pub mod lhd;
pub mod maximin_utils;
pub mod maxpro_utils;

// Python module definition
#[cfg(feature = "pyo3-bindings")]
#[pymodule]
fn _maxpro(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lhd::py_generate_lhd, m)?)?;
    m.add_function(wrap_pyfunction!(maxpro_utils::py_build_maxpro_lhd, m)?)?;
    m.add_function(wrap_pyfunction!(maxpro_utils::py_maxpro_criterion, m)?)?;
    m.add_function(wrap_pyfunction!(maximin_utils::py_build_maximin_lhd, m)?)?;
    m.add_function(wrap_pyfunction!(maximin_utils::py_maximin_criterion, m)?)?;
    m.add_function(wrap_pyfunction!(anneal::py_anneal_lhd, m)?)?;
    Ok(())
}
