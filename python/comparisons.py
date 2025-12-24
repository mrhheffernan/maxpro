import time

import numpy as np
import pyDOE3
import pymaxpro

import maxpro


def calculate_pymaxpro_criterion(maxpro_lhd: list[list[float]]) -> float:
    maxpro_inner = pymaxpro.maxpro_criterion(np.array(maxpro_lhd))
    n = len(maxpro_lhd)
    d = len(maxpro_lhd[0])
    true_maxpro = pymaxpro.calculate_true_maxpro(maxpro_inner, n, d)
    return true_maxpro


def test_parity():
    """
    Test parity between Python and Rust MaxPro criterion calculations
    """
    n_dims = [2, 4, 10]
    n_samples = [10, 100, 1000]
    iterations = 1000
    for nd in n_dims:
        for ns in n_samples:
            print(f"Checking {ns} samples in {nd} dimensions")
            maxpro_lhd = maxpro.build_maxpro_lhd(ns, nd, iterations)
            rust_criterion = maxpro.maxpro_criterion(maxpro_lhd)
            python_criterion = calculate_pymaxpro_criterion(maxpro_lhd)
            assert np.isclose(rust_criterion, python_criterion)


def benchmark_time_maxpro():
    """
    Benchmark rust vs. python implementation for time.
    """
    n_samples = 50
    n_iterations = 10000
    n_dim = 3

    time_rust_start = time.time()
    maxpro_lhd = maxpro.build_maxpro_lhd(n_samples, n_dim, n_iterations)
    maxpro_lhd = maxpro.anneal_lhd(maxpro_lhd, 1000, 1.0, 0.99, "maxpro", True)
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


def benchmark_time_maximin():
    n_dim = 5
    n_samples = 100
    n_iterations = 1000

    time_rust_start = time.time()
    maximin_lhd = maxpro.build_maximin_lhd(n_samples, n_dim, n_iterations)
    maximin_lhd = maxpro.anneal_lhd(maximin_lhd, 1000, 1.0, 0.99, "maximin", False)
    time_rust_end = time.time()

    time_python_start = time.time()
    pyDOE3_lhd = pyDOE3.lhs(n_dim, n_samples, "maximin", n_iterations)
    time_python_end = time.time()

    maxpro_maximin = maxpro.maximin_criterion(maximin_lhd)
    pyDOE3_maximin = maxpro.maximin_criterion(pyDOE3_lhd.tolist())

    rust_timer = time_rust_end - time_rust_start
    python_timer = time_python_end - time_python_start

    print(f"Rust criterion {maxpro_maximin} in {rust_timer} s")
    print(f"Python criterion {pyDOE3_maximin} in {python_timer} s")
    print(f"Python/Rust ratio: {python_timer / rust_timer}")


def main():
    print("Checking parity")
    test_parity()

    print("Benchmarking time between Python and Rust")
    benchmark_time_maxpro()

    print("Benchmarking Maximin time against reference implementation")
    benchmark_time_maximin()


if __name__ == "__main__":
    main()
