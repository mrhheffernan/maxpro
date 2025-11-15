use clap::{Parser, ValueEnum};
use maxpro::anneal::anneal_lhd;
use maxpro::lhd::plot_x_vs_y;
use maxpro::maximin_utils::{build_maximin_lhd, maximin_criterion};
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
    #[arg(short, long, default_value_t = 100000)]
    anneal_iterations: usize,
    #[arg(short, long, default_value_t = 0.99)]
    anneal_cooling: f64,
    #[arg(short, long, default_value_t = 1.0)]
    anneal_t: f64,
}

fn main() {
    let args: Args = Args::parse();
    let n_samples: usize = args.samples;
    let n_iterations: usize = args.iterations;
    let n_dims: usize = args.ndims;
    let plot: bool = args.plot;
    let metric = args.metric;
    let annealing_iterations = args.anneal_iterations;
    let annealing_t = args.anneal_t;
    let annealing_cooling = args.anneal_cooling;

    let lhd: Vec<Vec<f64>> = match metric {
        Metrics::MaxPro => build_maxpro_lhd(n_samples, n_dims, n_iterations),
        Metrics::MaxiMin => build_maximin_lhd(n_samples, n_dims, n_iterations),
    };

    let metric_value = match metric {
        Metrics::MaxPro => maxpro_criterion(&lhd),
        Metrics::MaxiMin => maximin_criterion(&lhd),
    };

    let annealed_design = match metric {
        Metrics::MaxPro => anneal_lhd(
            &lhd,
            annealing_iterations,
            annealing_t,
            annealing_cooling,
            |design: &Vec<Vec<f64>>| maxpro_criterion(design),
            true,
        ),
        Metrics::MaxiMin => anneal_lhd(
            &lhd,
            annealing_iterations,
            annealing_t,
            annealing_cooling,
            |design: &Vec<Vec<f64>>| maximin_criterion(design),
            false,
        ),
    };
    let annealed_metric = match metric {
        Metrics::MaxPro => maxpro_criterion(&annealed_design),
        Metrics::MaxiMin => maximin_criterion(&annealed_design),
    };

    // Outputs
    println!("{:?}", annealed_design);
    println!("Original metric: {metric_value}");
    println!("Annealed metric: {annealed_metric}");
    if plot {
        let _ = plot_x_vs_y(&annealed_design, std::path::Path::new(&args.output_path));
    }
}
