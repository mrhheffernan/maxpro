use clap::Parser;
use maxpro::anneal::anneal_lhd;
use maxpro::build_lhd;
use maxpro::enums::Metrics;
#[cfg(feature = "debug")]
use maxpro::lhd::plot_x_vs_y;
use maxpro::maximin_utils::maximin_criterion;
use maxpro::maxpro_utils::maxpro_criterion;
use rand::Rng;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    samples: u64,
    #[arg(short, long)]
    iterations: u64,
    #[arg(short, long)]
    plot: bool,
    #[arg(short, long)]
    ndims: u64,
    #[arg(short, long)]
    output_path: Option<String>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(short, long, value_enum, default_value_t = Metrics::MaxPro)]
    metric: Metrics,
    #[arg(long, default_value_t = 100000)]
    anneal_iterations: u64,
    #[arg(long, default_value_t = 0.99)]
    anneal_cooling: f64,
    #[arg(long, default_value_t = 1.0)]
    anneal_t: f64,
}

fn main() {
    // Set configurations from input parameters
    let args: Args = Args::parse();
    let n_samples: u64 = args.samples;
    let n_iterations: u64 = args.iterations;
    let n_dims: u64 = args.ndims;
    let plot: bool = args.plot;
    let metric = args.metric;
    let annealing_iterations = args.anneal_iterations;
    let annealing_t = args.anneal_t;
    let annealing_cooling = args.anneal_cooling;

    let seed = match args.seed {
        Some(x) => x,
        None => rand::random::<u64>(),
    };
    println!("Generating hypercube with seed {}", seed);

    // Ensure basic sanity checks are respected
    assert!(n_samples > 0, "n_samples must be positive and nonzero");
    assert!(
        n_iterations > 0,
        "n_iterations must be positive and nonzero"
    );
    assert!(n_dims > 0, "n_dims must be positive and nonzero");
    assert!(
        annealing_iterations > 0,
        "annealing_iterations must be positive and nonzero"
    );

    let (metric_fn, minimize) = match metric {
        Metrics::MaxPro => (maxpro_criterion as fn(&Vec<Vec<f64>>) -> f64, true),
        Metrics::MaxiMin => (maximin_criterion as fn(&Vec<Vec<f64>>) -> f64, false),
    };

    // Construct the initial latin hypercube
    let lhd: Vec<Vec<f64>> = build_lhd(n_samples, n_dims, n_iterations, Some(metric), &seed);

    let metric_value = metric_fn(&lhd);
    // Optimize the metric
    let annealed_design = anneal_lhd(
        &lhd,
        annealing_iterations,
        annealing_t,
        annealing_cooling,
        metric_fn,
        minimize,
        seed + 1,
    );
    let annealed_metric = metric_fn(&annealed_design);

    // Outputs
    println!("{:?}", annealed_design);
    println!("Original metric: {metric_value}");
    println!("Annealed metric: {annealed_metric}");

    // Plot, if requested
    if plot {
        if cfg!(feature = "debug") {
            if let Some(_output_path) = args.output_path {
                #[cfg(feature = "debug")]
                let _ = plot_x_vs_y(&annealed_design, std::path::Path::new(&_output_path));
            } else {
                println!("--output-path not specified, not writing to file.")
            }
        } else {
            println!("Plotting is only enabled with the debug feature flag.")
        }
    }
}
