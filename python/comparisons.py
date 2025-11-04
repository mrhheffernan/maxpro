import maxpro
import numpy as np
import time
import pymaxpro


def calculate_pymaxpro_criterion(maxpro_lhd):
    maxpro_inner = pymaxpro.maxpro_criterion(np.array(maxpro_lhd))
    n = len(maxpro_lhd)
    d = len(maxpro_lhd[0])
    true_maxpro = pymaxpro.calculate_true_maxpro(maxpro_inner, n, d)
    return true_maxpro


def main():
    n_samples = 50
    n_iterations = 100000
    n_dim = 2
    plot = False
    output_path = "."

    time_rust_start = time.time()
    maxpro_lhd = maxpro.build_maxpro_lhd(
        n_samples, n_iterations, n_dim, plot, output_path
    )
    maxpro_criterion = maxpro.maxpro_criterion(maxpro_lhd)
    time_rust_end = time.time()
    pymaxpro_lhd = pymaxpro.generate_maxpro_lhd_greedy(n_samples, n_dim, n_iterations)
    pymaxpro_criterion = calculate_pymaxpro_criterion(pymaxpro_lhd)
    time_py_end = time.time()

    rust_timer = time_rust_end - time_rust_start
    python_timer = time_py_end - time_rust_end

    print(f"Rust calculation: {maxpro_criterion} in {rust_timer} s")
    print(f"Python calculation: {pymaxpro_criterion} in {python_timer} s")
    print(f"Python / Rust ratio: {python_timer / rust_timer}")


if __name__ == "__main__":
    main()
