#[cfg(feature = "debug")]
use plotters::prelude::*;
#[cfg(feature = "pyo3-bindings")]
use pyo3::PyResult;
#[cfg(feature = "pyo3-bindings")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;

#[cfg(feature = "debug")]
/// A simple function to plot a scatter plot of the first two dimensions
/// of a Vec<Vec<f64>> for diagnostics.
///
/// Arguments:
///     data (&Vec<Vec<f64>>): Points to plot
///     output_path (&std::path::Path): Where to save the figure
///
/// Raises:
///     Err if data has fewer than 2 columns or 1 sample
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

/// Generates an LHD by taking in a number of samples and a number of dimensions
/// (parameters). This then creates a non-centered latin hypercube design.
///
/// Arguments:
///     n_samples (u64): Number of samples for the LHD
///     n_dim (u64): Number of dimensions for the LHD
///
/// Returns:
///     Vec<Vec<f64>>: A latin hypercube design
///
/// Panics:
///     n_samples == 0: n_samples must be positive and nonzero
///     n_dim == 0: n_dim must be positive and nonzero
pub fn generate_lhd(n_samples: u64, n_dim: u64, rng: &mut StdRng) -> Vec<Vec<f64>> {
    assert!(
        n_samples > 0,
        "n_samples must be a positive, nonzero integer"
    );

    assert!(
        n_samples <= usize::MAX as u64,
        "n_samples too large to index"
    );
    assert!(n_dim > 0, "n_dim must be a positive, nonzero integer");
    assert!(n_dim <= usize::MAX as u64, "n_dim too large to index");

    let n_samples_index: usize = n_samples.try_into().unwrap();
    let n_dim_index: usize = n_dim.try_into().unwrap();

    // initialize empty lhd
    let mut lhd: Vec<Vec<f64>> = vec![vec![0.0; n_dim_index]; n_samples_index];

    // For each dimension, start by generating a shuffle
    for j_idx in 0..n_dim_index {
        let mut permutation: Vec<usize> = (0..n_samples_index).collect();
        // Shuffle the 0..n_samples iterator
        permutation.shuffle(rng);

        for i_idx in 0..n_samples_index {
            // Get the range of the interval for this sample
            let interval_start: f64 = permutation[i_idx] as f64 / n_samples as f64;
            let interval_end: f64 = (permutation[i_idx] + 1) as f64 / n_samples as f64;
            // Generate a random sample in the box
            let sample: f64 = rng.random_range(interval_start..interval_end);
            // lhd[[i, j]] = sample;
            lhd[i_idx][j_idx] = sample;
        }
    }
    lhd
}

#[test]
/// Makes a simple test of a latin hypercube checking that,
/// for any interval in any dimension, there should be only one sample.
fn test_generate_lhd() {
    let n_samples: u64 = 100;
    let n_dim: u64 = 4;
    let seed: u64 = 12345;
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let design = generate_lhd(n_samples, n_dim, &mut rng);

    for dimension in 0..n_dim {
        // n_samples in an LHD is the same as the number of intervals to fill
        let mut counts = vec![0; n_samples.try_into().unwrap()];
        for sample_idx in 0..n_samples {
            // back-engineer the interval
            let sample: f64 = design[sample_idx as usize][dimension as usize];
            let sample_interval_idx = (sample * n_samples as f64).floor() as usize;
            counts[sample_interval_idx] += 1;
        }
        assert!(counts.iter().all(|&c| c == 1))
    }
}

#[test]
#[should_panic]
/// Ensure for 0 samples, generate_lhd panics.
fn test_generate_lhd_0_samples() {
    let n_samples: u64 = 0;
    let n_dim: u64 = 4;
    let seed: u64 = 12345;
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    generate_lhd(n_samples, n_dim, &mut rng);
}

#[test]
#[should_panic]
/// Ensure for 0 dimensions, generate_lhd panics.
fn test_generate_lhd_0_dimension() {
    let n_samples: u64 = 10;
    let n_dim: u64 = 0;
    let seed: u64 = 12345;
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    generate_lhd(n_samples, n_dim, &mut rng);
}

#[cfg(feature = "pyo3-bindings")]
/// Generates an unoptimized latin hypercube design
///
/// Args:
///     n_samples (int): Number of samples
///     n_dim (int): Number of dimensions
///     seed (int, optional): Seed for the random number generator
///
/// Returns:
///     list[list[float]]: Latin hypercube design
#[pyfunction(name = "generate_lhd")]
pub fn py_generate_lhd(n_samples: u64, n_dim: u64, seed: Option<u64>) -> PyResult<Vec<Vec<f64>>> {
    if n_samples == 0 {
        return Err(PyValueError::new_err(
            "n_samples must be a positive, nonzero integer",
        ));
    }
    if n_dim == 0 {
        return Err(PyValueError::new_err(
            "n_dim must be a positive, nonzero integer",
        ));
    }
    if n_samples > usize::MAX as u64 {
        return Err(PyValueError::new_err("n_samples too large to index"));
    }
    if n_dim > usize::MAX as u64 {
        return Err(PyValueError::new_err("n_dim too large to index"));
    }
    let rng_seed = match seed {
        Some(x) => x,
        None => rand::random::<u64>(),
    };
    let mut rng: StdRng = SeedableRng::seed_from_u64(rng_seed);
    Ok(generate_lhd(n_samples, n_dim, &mut rng))
}
