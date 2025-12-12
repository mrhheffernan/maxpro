import matplotlib.pyplot as plt
import numpy as np

import maxpro


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
    projected_metrics = {"maxpro_metric": [], "maximin_metric": []}
    for i in range(n_dimensions):
        projected_design = lhd[:, : -(i + 1)].tolist()
        projected_metric = maxpro.maxpro_criterion(projected_design)
        projected_metrics["maxpro_metric"].append([i + 1, projected_metric])

        projected_metric = maxpro.maximin_criterion(projected_design)
        projected_metrics["maximin_metric"].append([i + 1, projected_metric])

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
        plt.plot(metrics[:, 0], metrics[:, 1])
        plt.ylabel(metric)
        plt.xlabel("Projection dimension")
        plt.savefig(f"./projection_{metric}.png")


if __name__ == "__main__":
    main()
