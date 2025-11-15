use clap::{Parser, ValueEnum};
use maxpro::anneal::anneal_maxprolhd;
use maxpro::lhd::plot_x_vs_y;
use maxpro::maximin_utils::build_maximin_lhd;
use maxpro::maxpro_utils::{build_maxpro_lhd, maxpro_criterion};

#[derive(ValueEnum, Clone, Debug)]
enum Metrics {
    MaxPro,
    MaxiMin,
}

#[derive(Parser)]
struct Args {
    // The pattern to look for
    #[arg(short, long)]
    samples: usize,
    #[arg(short, long)]
    iterations: usize,
    #[arg(short, long)]
    plot: bool,
    #[arg(short, long)]
    ndims: usize,
    #[arg(short, long)]
    output_path: String,
    #[arg(short, long, value_enum, default_value_t = Metrics::MaxPro)]
    metric: Metrics,
}

fn main() {
    let args: Args = Args::parse();
    let n_samples: usize = args.samples;
    let n_iterations: usize = args.iterations;
    let n_dims: usize = args.ndims;
    let plot: bool = args.plot;
    let metric = args.metric;

    let lhd: Vec<Vec<f64>> = match metric {
        Metrics::MaxPro => build_maxpro_lhd(n_samples, n_dims, n_iterations),
        Metrics::MaxiMin => build_maximin_lhd(n_samples, n_dims, n_iterations),
    };
    println!("{:?}", lhd);
    let metric_value = maxpro_criterion(&lhd);
    // all below are dummy values for now
    let n_iterations: usize = 100000;
    let initial_temp: f64 = 2.0;
    let cooling_rate: f64 = 0.999;
    let annealed_design = anneal_maxprolhd(&lhd, n_iterations, initial_temp, cooling_rate);
    let annealed_metric = maxpro_criterion(&annealed_design);

    // Outputs
    println!("Original metric: {metric_value}");
    println!("Annealed metric: {annealed_metric}");
    if plot {
        let _ = plot_x_vs_y(&annealed_design, std::path::Path::new(&args.output_path));
    }
}
