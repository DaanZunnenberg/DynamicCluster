"""Entry point: run the simulation-estimation study and write results to a CSV file.

Configure the study by editing the constants at the top of `main()`, then run:

    python scripts/run_simulation.py
"""

import sys
import csv
import concurrent.futures
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamiccluster import (SimulationState,
                            initialize_simulation_matrices,
                            simulate_data,
                            estimate_maximum_likelihood)
from dynamiccluster.simulation import (N_UNITS_PER_CLUSTER, N_FEATURES, N_CLUSTERS,
                                       INVERSE_DEGREES_OF_FREEDOM, N_TIME_STEPS,
                                       CORRELATION_MATRIX, CIRCLE_MIDPOINT_DISTANCE,
                                       CIRCLE_RADIUS, N_LAPS, SMOOTHING,
                                       TIME_VARYING_PROBABILITIES, GAMMA,
                                       SEMI_MAJOR_AXIS, ELLIPSE_1_ORIENTATION,
                                       ELLIPSE_2_ORIENTATION, SIMULATION_TYPE)
from dynamiccluster.estimation import (N_CLUSTERS as EST_N_CLUSTERS,
                                       N_STARTING_VALUES, VERBOSITY,
                                       INVERSE_DEGREES_OF_FREEDOM as EST_INVERSE_DEGREES_OF_FREEDOM,
                                       SMOOTHING as EST_SMOOTHING,
                                       MEAN_SPECIFIC_SMOOTHING, WARM_START,
                                       REGULARIZE_COVARIANCE, BURN_IN)


def run_one_simulation(simulation_index, state, simulation_params, estimation_params):
    """Simulate one data set and estimate the model on it. Used for parallel runs."""
    print('\n============================')
    print(f'Curent simulation : {simulation_index}')
    print('============================\n')

    simulate_data(state, simulation_params)
    return estimate_maximum_likelihood(state, estimation_params)


def main():
    #########################################
    # Study configuration
    #########################################
    random_seed = 1234
    n_simulations = 250
    results_path_prefix = "results"
    verbosity = 0
    run_in_parallel = False
    simulation_type = 3             # 0 for ovals
                                    # 1 for converging circles
                                    # 2 for Y-shaped
                                    # 3 for flipped-Y.

    # Cluster arguments.
    inverse_degrees_of_freedom_simulation = 0
    inverse_degrees_of_freedom_estimation = 0
    n_clusters_simulation = 2
    n_clusters_estimation = 2
    circle_radius = 2
    circle_midpoint_distance = 2
    n_laps = 1

    # Cluster dimension arguments.
    n_time_steps = 40
    n_features = 2
    n_units_per_cluster = 100  # total number of units is n_units_per_cluster * n_clusters_simulation

    gamma = 0.25
    time_varying_probabilities = 1  # if 0, transition probs are constant;
                                    # if 1, they are time-varying.
                                    # This alters the trajectories of the true means.
    # Geometric arguments.
    semi_major_axis = 2
    simulation_variance_1 = 1
    simulation_variance_2 = 1
    covariance = 0.9
    ellipse1_orientation = 1        # 1 for standing
    ellipse2_orientation = 0        # 0 for lying down
    smoothing_estimation = 0.25
    smoothing_dgp = 0.1

    # Estimation-related arguments.
    n_starting_values = 1
    warm_start = 1.7
    regularize_covariance = 0
    mean_specific_smoothing = 0  # if 1, a smoothing parameter per mean;
                                 # else a single smoothing parameter for all.
    burn_in = 0

    #########################################
    # Initialization
    #########################################
    correlation_matrix = np.array([[simulation_variance_1, -covariance],
                                   [-covariance, simulation_variance_2]])

    simulation_params = {
        N_UNITS_PER_CLUSTER: n_units_per_cluster,
        N_FEATURES: n_features,
        N_CLUSTERS: n_clusters_simulation,
        INVERSE_DEGREES_OF_FREEDOM: inverse_degrees_of_freedom_simulation,
        N_TIME_STEPS: n_time_steps,
        CORRELATION_MATRIX: correlation_matrix,
        CIRCLE_MIDPOINT_DISTANCE: circle_midpoint_distance,
        CIRCLE_RADIUS: circle_radius,
        N_LAPS: n_laps,
        SMOOTHING: smoothing_dgp,
        TIME_VARYING_PROBABILITIES: time_varying_probabilities,
        GAMMA: gamma,
        SEMI_MAJOR_AXIS: semi_major_axis,
        ELLIPSE_1_ORIENTATION: ellipse1_orientation,
        ELLIPSE_2_ORIENTATION: ellipse2_orientation,
        SIMULATION_TYPE: simulation_type,
    }

    estimation_params = {
        EST_N_CLUSTERS: n_clusters_estimation,
        N_STARTING_VALUES: n_starting_values,
        VERBOSITY: verbosity,
        EST_INVERSE_DEGREES_OF_FREEDOM: inverse_degrees_of_freedom_estimation,
        EST_SMOOTHING: smoothing_estimation,
        MEAN_SPECIFIC_SMOOTHING: mean_specific_smoothing,
        WARM_START: warm_start,
        REGULARIZE_COVARIANCE: regularize_covariance,
        BURN_IN: burn_in,
    }

    # Allocate the data/parameter matrices shared across simulation and estimation.
    state = SimulationState()

    (state.estimated_parameters,
     state.true_parameters,
     state.true_states,
     state.predicted_cluster_probabilities,
     state.filtered_cluster_probabilities,
     state.data) = initialize_simulation_matrices(n_time_steps,
                                                    n_units_per_cluster,
                                                    n_features,
                                                    n_clusters_estimation,
                                                    n_clusters_simulation)

    #########################################
    # Simulation and estimation
    #########################################
    np.random.seed(random_seed)
    print(f'We are setting the random seed to {random_seed}.')

    if run_in_parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            tasks = [executor.submit(run_one_simulation, sim_index, state, simulation_params, estimation_params)
                      for sim_index in range(n_simulations)]

        results = [task.result() for task in tasks]
        results = np.vstack(results)
    else:
        results = []
        for sim_index in range(n_simulations):
            print('\n============================')
            print(f'Curent simulation : {sim_index}')
            print('============================\n')

            simulate_data(state, simulation_params)
            results.append(estimate_maximum_likelihood(state, estimation_params))
        results = np.vstack(results).T

    # Build the output file name from the study configuration.
    file_name = f"Sim_{results_path_prefix}_{simulation_type}_{n_units_per_cluster}_{n_time_steps}_{gamma}_\
{circle_radius}_{circle_midpoint_distance}_\
{inverse_degrees_of_freedom_simulation}_\
{inverse_degrees_of_freedom_estimation}_\
{n_clusters_estimation}_\
{n_clusters_simulation}_\
{time_varying_probabilities}_{mean_specific_smoothing}_{semi_major_axis}_\
{ellipse1_orientation}_{ellipse2_orientation}\
_{simulation_variance_1}_{simulation_variance_2}_{covariance}_\
{smoothing_estimation}_{smoothing_dgp}.csv"

    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(results)

    print("\n============================")
    print("END OF PROGRAM")
    print("============================\n")
    print(f"\n Results matrix is stored under the name:\n {file_name}")


if __name__ == "__main__":
    main()
