use clap::Parser;
use maxpro::utils::build_maxpro_lhd;

#[derive(Parser)]
struct Args {
    // The pattern to look for
    #[arg(short, long)]
    samples: i32,
    #[arg(short, long)]
    iterations: i32,
    #[arg(short, long)]
    plot: bool,
    #[arg(short, long)]
    ndims: usize,
}

fn main() {
    let args = Args::parse();
    let n_samples: i32 = args.samples;
    let n_iterations = args.iterations;
    let n_dims = args.ndims;
    let plot = args.plot;
    let maxpro_lhd = build_maxpro_lhd(n_samples, n_iterations, n_dims, plot);
    println!("{:?}", maxpro_lhd)
}
