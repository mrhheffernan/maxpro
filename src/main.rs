use clap::Parser;
use maxpro::utils::{build_maxpro_lhd, plot_x_vs_y};

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
}

fn main() {
    let args: Args = Args::parse();
    let n_samples: usize = args.samples;
    let n_iterations: usize = args.iterations;
    let n_dims: usize = args.ndims;
    let plot: bool = args.plot;

    let maxpro_lhd: Vec<Vec<f64>> = build_maxpro_lhd(n_samples, n_dims, n_iterations);
    if plot {
        let _ = plot_x_vs_y(&maxpro_lhd, std::path::Path::new(&args.output_path));
    }
    println!("{:?}", maxpro_lhd)
}
