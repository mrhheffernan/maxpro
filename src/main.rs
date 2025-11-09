use clap::{Parser, ValueEnum};
use maxpro::lhd::plot_x_vs_y;
use maxpro::maximin_utils::build_maximin_lhd;
use maxpro::maxpro_utils::build_maxpro_lhd;

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

    if plot {
        let _ = plot_x_vs_y(&lhd, std::path::Path::new(&args.output_path));
    }
    println!("{:?}", lhd)
}
