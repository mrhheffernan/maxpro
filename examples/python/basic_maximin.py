import maxpro

def main():
    # Fix the randomness
    SEED = 42

    # Set number of samples, dimensions, and iterations
    n_samples = 100
    n_dim = 3
    n_iterations = 500

    # Use the maximum projection metric
    metric = 'maximin'

    # Generate a semi-optimal latin hypercube design
    lhd = maxpro.build_lhd(n_samples, n_dim, n_iterations, metric, SEED)
    metric_value = maxpro.maximin_criterion(lhd)
    print(f"Initial maximin metric value: {metric_value}")

    # Use annealing to further minimize the metric value
    n_anneal_iterations = 5000
    initial_temperature = 1
    cooling_rate = 0.99
    minimize = False

    lhd_annealed = maxpro.anneal_lhd(lhd, n_anneal_iterations, initial_temperature, cooling_rate, metric, minimize, SEED)

    metric_value_annealed = maxpro.maximin_criterion(lhd_annealed)
    print(f"Annealed maximin metric value: {metric_value_annealed}")

if __name__ == "__main__":
    main()