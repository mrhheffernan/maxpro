import math
import random
import time

import numpy as np
from scipy.stats import qmc


def maxpro_criterion(design: np.ndarray) -> float:
    """
    Calculates the internal sum term of the MaxPro criterion (psi(D)).
    This sum is the term that is directly minimized in the optimization
    process.

    The minimized term is: sum_{i<j} [ 1 / product_{l=1}^{d} (x_il - x_jl)^2 ]

    Args:
        design: A NumPy array (n x d) representing the design points,
                normalized to the unit hypercube [0, 1]^d.

    Returns:
        The value of the internal sum (the MaxPro Sum Metric).
    """
    n, d = design.shape

    if n < 2:
        return 0.0

    inverse_product_sum = 0.0
    epsilon = 1e-12  # Small constant to prevent division by zero

    for i in range(n):
        for j in range(i + 1, n):
            # Calculate the product term: product_{l=1}^{d} (x_il - x_jl)^2
            diffs = design[i, :] - design[j, :]
            product_of_squared_diffs = np.prod(diffs**2) + epsilon

            # Sum the inverse products
            inverse_product_sum += 1.0 / product_of_squared_diffs

    return inverse_product_sum


def calculate_true_maxpro(
    maxpro_sum_metric: float, n_points: int, n_dims: int
) -> float:
    """
    Calculates the final MaxPro Criterion (psi(D)) from the minimized sum.

    psi(D) = (1 / (n choose 2) * MaxPro_Sum_Metric)^(1/d)

    Args:
        maxpro_sum_metric: The result of the maxpro_criterion function.
        n_points: The number of design points (n).
        n_dims: The number of dimensions (d or p).

    Returns:
        The true MaxPro Criterion value.
    """
    if n_points < 2 or n_dims == 0:
        return 0.0

    n_pairs = n_points * (n_points - 1) / 2.0
    return (maxpro_sum_metric / n_pairs) ** (1.0 / n_dims)


def generate_maxpro_lhd_greedy(
    n_points: int, n_dims: int, max_iterations: int = 50000
) -> np.ndarray:
    """
    Generates a MaxPro Latin Hypercube Design using a simple iterative random
    swap heuristic (Greedy search).

    Args:
        n_points: The number of design points (n).
        n_dims: The number of dimensions (d or p).
        max_iterations: The number of optimization attempts (random swaps).

    Returns:
        The optimized design matrix (n x d) on the unit hypercube [0, 1]^d.
    """
    print("--- Starting Greedy LHD Optimization ---")
    print(f"N (Points): {n_points}, D (Dimensions): {n_dims}, ITERS: {max_iterations}")

    # 1. Generate an initial, standard Latin Hypercube Design (LHD)
    sampler = qmc.LatinHypercube(d=n_dims)
    best_design = sampler.random(n=n_points)
    best_metric = maxpro_criterion(best_design)

    print(f"Initial MaxPro Sum Metric: {best_metric:,.4f}")

    start_time = time.time()

    # 2. Optimization Loop
    for iteration in range(max_iterations):

        # Generate a new candidate
        candidate_design = sampler.random(n=n_points)
        candidate_metric = maxpro_criterion(candidate_design)

        if candidate_metric < best_metric:
            best_design = candidate_design
            best_metric = candidate_metric

        if (iteration + 1) % (max_iterations // 10) == 0:
            print(
                f"Iteration {iteration + 1}/{max_iterations}: Best MaxPro Sum = {best_metric:,.4f}"
            )

    end_time = time.time()
    true_maxpro_value = calculate_true_maxpro(best_metric, n_points, n_dims)

    print("--- Greedy Optimization Complete ---")
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    print(f"Final MaxPro Sum Metric: {best_metric:,.4f}")
    print(f"True MaxPro Criterion (psi(D)): {true_maxpro_value:.6f}")

    return best_design


# --- Example Usage ---

if __name__ == "__main__":
    # Define design parameters
    N = 20  # Number of points
    D = 4  # Number of dimensions
    ITERS = 50000  # Optimization iterations

    optimized_lhd_greedy = generate_maxpro_lhd_greedy(N, D, ITERS)

    print("\n--- Results for Greedy Optimization ---")
    print(f"Final MaxPro Sum Metric: {maxpro_criterion(optimized_lhd_greedy):,.4f}")
