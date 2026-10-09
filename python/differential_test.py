import time

import maxpro
import numpy as np
import pydoe

SEED = 42


def test_parity():
    """
    Test parity between Python and Rust MaxPro criterion calculations
    """
    n_dims = [2, 4, 10]
    n_samples = [10, 100, 1000]
    iterations = 1000
    seed = SEED
    for nd in n_dims:
        for ns in n_samples:
            print(f"Checking {ns} samples in {nd} dimensions")
            maxpro_lhd = maxpro.build_lhd(ns, nd, iterations, "maxpro", seed)
            rust_criterion = maxpro.maxpro_criterion(maxpro_lhd)
            python_criterion = pydoe.space_filling.stochastic.maxpro._maxpro_criterion(
                np.asarray(maxpro_lhd)
            )
            ## Cannot be compared directly since maxpro calculates the full criterion and pydoe only calculates the inner inverse sum
            assert np.isclose(rust_criterion, python_criterion), (
                f"maxpro: {rust_criterion}, pydoe: {python_criterion}"
            )
            seed += 1


def benchmark_time_maxpro():
    """
    Benchmark rust vs. python implementation for time.
    """
    n_samples = 100
    n_iterations = 10000
    n_dim = 3

    time_rust_start = time.time()
    maxpro_lhd = maxpro.build_lhd(n_samples, n_dim, 1000, "maxpro", SEED)
    # Spend iterations primarily on annealing
    maxpro_lhd = maxpro.anneal_lhd(
        maxpro_lhd, n_iterations, 1.0, 0.99, "maxpro", True, True, SEED
    )
    maxpro_criterion = maxpro.maxpro_criterion(maxpro_lhd)
    time_rust_end = time.time()

    pymaxpro_lhd = pydoe.maxpro_design(
        n_samples, n_dim, iterations=n_iterations, seed=SEED
    )
    pymaxpro_criterion_inner_sum = (
        pydoe.space_filling.stochastic.maxpro._maxpro_criterion(pymaxpro_lhd)
    )
    time_py_end = time.time()
    pymaxpro_criterion = maxpro.maxpro_criterion(pymaxpro_lhd.tolist())

    rust_timer = time_rust_end - time_rust_start
    python_timer = time_py_end - time_rust_end

    print(f"Rust calculation: {maxpro_criterion} in {rust_timer} s")
    print(f"Python calculation: {pymaxpro_criterion} in {python_timer} s")
    print(f"Python / Rust ratio: {python_timer / rust_timer}")


def main():
    # print("Checking parity")
    # test_parity()

    print("Benchmarking time between Python and Rust")
    benchmark_time_maxpro()


if __name__ == "__main__":
    main()
