import math
import time

import maxpro

R_SNIPPET = """install.packages("MaxPro")
library("MaxPro")
n <- 100
p <- 2
design <- MaxProLHD(n, p, s=2, temp0=0, nstarts = 1, itermax = 400, total_iter = 1e+06)
write.table(design$Design, col.names=FALSE, row.names=FALSE)
"""

R_DESIGN = """0.005 0.145
0.015 0.685
0.025 0.395
0.035 0.845
0.045 0.255
0.055 0.525
0.065 0.095
0.075 0.765
0.085 0.455
0.095 0.215
0.105 0.625
0.115 0.915
0.125 0.345
0.135 0.715
0.145 0.055
0.155 0.485
0.165 0.805
0.175 0.285
0.185 0.565
0.195 0.975
0.205 0.185
0.215 0.665
0.225 0.435
0.235 0.885
0.245 0.015
0.255 0.605
0.265 0.365
0.275 0.735
0.285 0.125
0.295 0.505
0.305 0.835
0.315 0.235
0.325 0.645
0.335 0.935
0.345 0.035
0.355 0.545
0.365 0.315
0.375 0.795
0.385 0.165
0.395 0.995
0.405 0.695
0.415 0.385
0.425 0.865
0.435 0.585
0.445 0.065
0.455 0.955
0.465 0.265
0.475 0.755
0.485 0.475
0.495 0.115
0.505 0.895
0.515 0.355
0.525 0.655
0.535 0.195
0.545 0.815
0.555 0.415
0.565 0.005
0.575 0.615
0.585 0.295
0.595 0.925
0.605 0.515
0.615 0.085
0.625 0.725
0.635 0.245
0.645 0.985
0.655 0.445
0.665 0.155
0.675 0.785
0.685 0.555
0.695 0.045
0.705 0.675
0.715 0.335
0.725 0.875
0.735 0.105
0.745 0.595
0.755 0.405
0.765 0.745
0.775 0.225
0.785 0.945
0.795 0.495
0.805 0.135
0.815 0.825
0.825 0.305
0.835 0.635
0.845 0.025
0.855 0.465
0.865 0.905
0.875 0.175
0.885 0.375
0.895 0.775
0.905 0.275
0.915 0.535
0.925 0.965
0.935 0.075
0.945 0.705
0.955 0.205
0.965 0.425
0.975 0.855
0.985 0.325
0.995 0.575"""

R_MEASURE = 95.98506
R_TIME_SECONDS = 5.503749
R_ITERATIONS = 63401


def process_r() -> list[list[float]]:
    design_str = [line.split(" ") for line in R_DESIGN.split("\n")]
    lhd = [[float(d) for d in line] for line in design_str]
    return lhd


def main():
    # Compare metric calculation
    r_lhd = process_r()
    maxpro_criterion = maxpro.maxpro_criterion(r_lhd)
    print(f"R calculation: {R_MEASURE} \nRust calculation: {maxpro_criterion}")
    assert math.isclose(maxpro_criterion, R_MEASURE, rel_tol=1e-7), (
        "Calculations differ between Rust and R"
    )

    # Compare runtimes
    SEED = 42
    start_time = time.time()
    maxpro_lhd = maxpro.build_lhd(100, 2, R_ITERATIONS, "maxpro", SEED)
    optimal_maxpro_lhd = maxpro.anneal_lhd(
        maxpro_lhd, int(R_ITERATIONS / 2), 1, 0.5, "maxpro", True, SEED
    )
    end_time = time.time()
    duration_seconds = end_time - start_time
    metric_value = maxpro.maxpro_criterion(optimal_maxpro_lhd)
    print(f"Design with metric {metric_value} found in {duration_seconds} seconds")
    print(f"Rust/R runtime ratio: {duration_seconds / R_TIME_SECONDS}")


if __name__ == "__main__":
    main()
