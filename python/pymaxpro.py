import numpy as np
from scipy.stats import qmc
import random
import time
import math


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


def optimize_maxpro_sa(
    initial_design: np.ndarray,
    max_iterations: int = 50000,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.999,
) -> np.ndarray:
    """
    Optimizes a candidate Latin Hypercube Design (LHD) for the MaxPro metric
    using a Simulated Annealing (SA) approach.

    Args:
        initial_design: A NumPy array (n x d) representing the starting LHD.
        max_iterations: The number of temperature steps/swaps.
        initial_temp: Starting temperature for the SA schedule.
        cooling_rate: Multiplicative factor for geometric cooling (T = T * rate).

    Returns:
        The optimized design matrix (n x d) on the unit hypercube [0, 1]^d.
    """
    n_points, n_dims = initial_design.shape
    print("\n--- Starting Simulated Annealing Optimization ---")
    print(f"N (Points): {n_points}, D (Dimensions): {n_dims}, ITERS: {max_iterations}")
    print(f"Initial Temp: {initial_temp}, Cooling Rate: {cooling_rate}")

    current_design = initial_design.copy()
    current_metric = maxpro_criterion(current_design)
    best_design = current_design.copy()
    best_metric = current_metric

    temp = initial_temp
    start_time = time.time()

    for iteration in range(max_iterations):

        # 1. Propose a new state (random LHD preserving swap)
        candidate_design = current_design.copy()

        # Choose a dimension (column) and two different points (rows) to swap
        dim_to_swap = random.randrange(n_dims)
        i, j = random.sample(range(n_points), 2)

        # Perform the swap (maintains the LHD property)
        candidate_design[i, dim_to_swap], candidate_design[j, dim_to_swap] = (
            candidate_design[j, dim_to_swap],
            candidate_design[i, dim_to_swap],
        )

        # 2. Evaluate the new state
        candidate_metric = maxpro_criterion(candidate_design)

        # Calculate the change in metric (Minimization: delta < 0 is an improvement)
        delta_metric = candidate_metric - current_metric

        # 3. Acceptance criterion
        # If the new state is better (lower metric), accept it
        if delta_metric < 0:
            accept_probability = 1.0
        # If the new state is worse, calculate acceptance probability
        else:
            # P(accept) = exp(-delta_metric / T)
            # We use max(1e-10, temp) to prevent division by zero in case temp hits zero
            accept_probability = math.exp(-delta_metric / max(1e-10, temp))

        # 4. Metropolis condition: Accept the candidate based on probability
        if random.random() < accept_probability:
            current_design = candidate_design
            current_metric = candidate_metric

            # Update overall best design if current accepted state is the best so far
            if current_metric < best_metric:
                best_metric = current_metric
                best_design = current_design.copy()

        # 5. Cool the system (geometric cooling)
        temp *= cooling_rate

        if (iteration + 1) % (max_iterations // 10) == 0:
            print(
                f"Iteration {iteration + 1}/{max_iterations}: Best MaxPro Sum = {best_metric:,.4f}, Current Temp = {temp:.4e}"
            )

    end_time = time.time()
    true_maxpro_value = calculate_true_maxpro(best_metric, n_points, n_dims)

    print("--- SA Optimization Complete ---")
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

    # ---------------------------------------------
    # 1. Greedy Optimization Example (Renamed existing function)
    # ---------------------------------------------
    optimized_lhd_greedy = generate_maxpro_lhd_greedy(N, D, ITERS)

    print("\n--- Results for Greedy Optimization ---")
    print(f"Final MaxPro Sum Metric: {maxpro_criterion(optimized_lhd_greedy):,.4f}")

    # ---------------------------------------------
    # 2. Simulated Annealing Example
    # ---------------------------------------------

    optimized_lhd_sa = optimize_maxpro_sa(
        optimized_lhd_greedy,
        max_iterations=ITERS,
        initial_temp=0.1,  # A lower initial temp may be suitable for small N
        cooling_rate=0.9995,  # A slightly slower cooling rate
    )

    print("\n--- Results for Simulated Annealing Optimization ---")
    final_sa_sum_metric = maxpro_criterion(optimized_lhd_sa)
    final_sa_true_maxpro = calculate_true_maxpro(final_sa_sum_metric, N, D)

    print(f"Final MaxPro Sum Metric: {final_sa_sum_metric:,.4f}")
    print(f"True MaxPro Criterion (psi(D)): {final_sa_true_maxpro:.6f}")

    # Example of scaling the best SA design
    min_bounds = np.array([10.0, 50.0, 0.0, 1.0])
    max_bounds = np.array([20.0, 100.0, 1.0, 5.0])
    scaled_lhd_sa = min_bounds + (max_bounds - min_bounds) * optimized_lhd_sa

    print("\nScaled LHD from Simulated Annealing (first 5 points):")
    print(scaled_lhd_sa[:5])
    print(f"Shape: {scaled_lhd_sa.shape}")
