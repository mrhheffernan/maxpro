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
    
    let metric_value = match metric {
        Metrics::MaxPro => maxpro_criterion(&lhd),
        Metrics::MaxiMin => maximin_criterion(&lhd),
    };
    // all below are dummy values for now
    let n_iterations: usize = 100000;
    let initial_temp: f64 = 1.0;
    let cooling_rate: f64 = 0.999;
    let annealed_design = match metric {
        Metrics::MaxPro => anneal_lhd(
            &lhd,
            n_iterations,
            initial_temp,
            cooling_rate,
            |design: &Vec<Vec<f64>>| maxpro_criterion(design),
            true,
        ),
        Metrics::MaxiMin => anneal_lhd(
            &lhd,
            n_iterations,
            initial_temp,
            cooling_rate,
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
