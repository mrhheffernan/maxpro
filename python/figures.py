from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

import maxpro

# Small buffer for stability
EPSILON = 1e-5


def generalized_maximin_measure(design, q):
    """
    Calculates the generalized maximin measure (Mm_q) for a design.

    Parameters
    ----------
    design: np.ndarray
        The design matrix (n_points x n_dimensions).
        n_points is 'n' in the formula.
    q: int
        The projection dimension.

    Returns
    -------
    float
        The Mm_q value.
    """

    n_points, n_dimensions = design.shape

    if q >= n_dimensions:
        # If q is equal to or larger than the full dimension, only the full
        # space needs to be considered (r=1)
        q = n_dimensions

    # The term 1 / C(n, 2)
    n_choose_2 = n_points * (n_points - 1) / 2
    if n_choose_2 == 0:
        return 0.0  # Handle case with 0 or 1 point

    term_prefactor = 1.0 / n_choose_2

    # 1. Identify all possible q-dimensional projections (combinations)
    # The 'r' index in the formula corresponds to these combinations.
    projection_indices = list(combinations(range(n_dimensions), q))

    # List to store the summation result for each projection
    projection_sums = []

    # Loop over every possible q-dimensional projection (r-th projection)
    for r_indices in projection_indices:
        # Extract the design points for the current projection
        # This is the "r-th projection of dimension q"
        projected_design = design[:, r_indices]

        # Calculate the pairwise squared Euclidean distances (d_qr^2)
        # Note: Scipy's pdist is often faster, but we use numpy for simplicity.

        # Calculate the difference matrix (x_i - x_j) for all pairs
        diff = projected_design[:, np.newaxis, :] - projected_design[np.newaxis, :, :]

        # Square the differences and sum across the q dimensions (axis=-1)
        # This gives the squared Euclidean distance: d_qr^2
        squared_distances = np.sum(diff**2, axis=-1)

        # The formula uses d_qr^(2q). Since we have d_qr^2, we raise it to the power of q.
        # d_qr^(2q) = (d_qr^2)^q
        power_2q_distances = squared_distances**q

        # 2. Perform the double summation
        # Sum only the upper triangle (i < j) to match the formula's i=1 to n-1, j=i+1 to n
        summation = 0.0

        for i in range(n_points):
            for j in range(i + 1, n_points):
                # Check for zero distance (degenerate case)
                if power_2q_distances[i, j] == 0:
                    return 0.0  # Mm_q would be 0 if points overlap

                # Sum the inverse distances raised to the power 2q
                summation += 1.0 / power_2q_distances[i, j]

        # Apply the prefactor and store the result for this projection
        projection_sums.append(term_prefactor * summation)

    # 3. Apply the maximin criterion: take the minimum result across all projections
    # The measure itself is the inverse of the final term raised to the power -1/(2q)
    if not projection_sums:
        return np.inf  # Should not happen unless n < 2

    min_sum = np.min(projection_sums)

    # 4. Final calculation: (min_sum)^(-1 / (2q))
    if min_sum <= 0:
        return EPSILON

    Mm_q = min_sum ** (-1.0 / (2.0 * q))

    return Mm_q


def calculate_maximin_projections(
    lhd: list[list[float]],
) -> dict[str, list[list[float]]]:
    """
    Calculate projection metric values

    :param lhd: Description
    :type lhd: list[list[float]]
    :return: Projection metric values by projection dimension
    :rtype: dict[str, list[list[float]]]
    """

    lhd = np.array(lhd)
    n_dimensions = lhd.shape[1]
    projected_metrics = {
        "maxpro_metric": [],
        "maximin_metric": [],
        "generalized_maximin": [],
    }
    for i in range(1, n_dimensions):
        projected_design = lhd[:, : -(i + 1)].tolist()
        projected_maxpro = maxpro.maxpro_criterion(projected_design)
        projected_metrics["maxpro_metric"].append([i + 1, projected_maxpro])

        projected_maximin = maxpro.maximin_criterion(projected_design)
        projected_metrics["maximin_metric"].append([i + 1, projected_maximin])

        projected_genmaximin = generalized_maximin_measure(lhd, i)
        projected_metrics["generalized_maximin"].append([i + 1, projected_genmaximin])

    return projected_metrics


def main():
    """Reproduce plots from the original MaxPro paper"""
    ndim = 10
    nsamples = 100
    niterations = 10000
    maxpro_lhd = maxpro.build_maxpro_lhd(nsamples, ndim, niterations)
    optimal_maxpro_lhd = maxpro.anneal_lhd(
        maxpro_lhd,
        1000,
        initial_temp=1.0,
        cooling_rate=0.1,
        metric_name="maxpro",
        minimize=True,
    )
    projected_metrics = calculate_maximin_projections(optimal_maxpro_lhd)

    for metric in projected_metrics:
        metrics = np.array(projected_metrics[metric])
        plt.figure()
        plt.plot(metrics[:, 0], metrics[:, 1], marker="^")
        plt.ylabel(metric)
        plt.xlabel("Projection dimension")
        plt.savefig(f"./projection_{metric}.png")


if __name__ == "__main__":
    main()
