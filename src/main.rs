use clap::Parser;
use maxpro::utils::build_maxpro_lhd;

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
    let mut output_path = std::string::String::new();
    if args.plot {
        // Assign the variable iff args.plot is used
        output_path = args.output_path;
    }
    let maxpro_lhd: Vec<Vec<f64>> =
        build_maxpro_lhd(n_samples, n_iterations, n_dims, plot, output_path);
    println!("{:?}", maxpro_lhd)
}
