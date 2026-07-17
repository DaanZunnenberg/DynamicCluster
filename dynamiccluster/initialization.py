"""Allocation of the arrays used throughout a simulation-estimation cycle."""

import numpy as np


def initialize_time_varying_parameter_structure(n_time_steps, n_rows, n_cols, fill_value=0):
    """Create a (n_time_steps, n_rows, n_cols) array filled with `fill_value`.

    Parameters
    ----------
    n_time_steps : int
        Number of time periods (may be passed as a float and is cast to int).
    n_rows : int
        Row dimension of each time slice (may be passed as a float, e.g. when
        computed from a formula, and is cast to int).
    n_cols : int
        Column dimension of each time slice (same casting rule as `n_rows`).
    fill_value : float, optional
        Value to fill every entry with. Defaults to 0.

    Returns
    -------
    ndarray, shape (n_time_steps, n_rows, n_cols)
    """
    n_time_steps = int(n_time_steps)
    n_rows = int(n_rows)
    n_cols = int(n_cols)

    return np.full((n_time_steps, n_rows, n_cols), fill_value, dtype=float)


def initialize_simulation_matrices(n_time_steps, n_units_per_cluster, n_features,
                                    n_clusters_estimation, n_clusters_simulation):
    """Allocate the core data/parameter arrays used in a simulation-estimation cycle.

    Parameters
    ----------
    n_time_steps : int
        Number of time periods.
    n_units_per_cluster : int
        Number of firms (units) per mixture component.
    n_features : int
        Number of firm characteristics (dimensionality of the data).
    n_clusters_estimation : int
        Number of mixture components assumed during estimation.
    n_clusters_simulation : int
        Number of mixture components used to simulate the data.

    Returns
    -------
    estimated_parameters : ndarray
        Parameters for estimation.
    true_parameters : ndarray
        True parameters used for simulation.
    true_states : ndarray
        True cluster states for simulation.
    predicted_cluster_probabilities : ndarray
        Predicted cluster probabilities for estimation.
    filtered_cluster_probabilities : ndarray
        Filtered cluster probabilities for estimation.
    data : ndarray
        Storage for the simulated data.
    """
    # Each cluster's parameters are n_features means plus the vech() of the
    # Cholesky factor of an n_features x n_features covariance matrix.
    n_params_per_cluster = 0.5 * n_features * (n_features + 3)

    estimated_parameters = initialize_time_varying_parameter_structure(
        n_time_steps, n_clusters_estimation, n_params_per_cluster)
    true_parameters = initialize_time_varying_parameter_structure(
        n_time_steps, n_clusters_simulation, n_params_per_cluster)
    true_states = np.zeros((n_time_steps, n_clusters_simulation * n_units_per_cluster))

    predicted_cluster_probabilities = initialize_time_varying_parameter_structure(
        n_time_steps, n_features * n_units_per_cluster, n_clusters_estimation,
        1.0 / n_clusters_estimation)
    filtered_cluster_probabilities = initialize_time_varying_parameter_structure(
        n_time_steps, n_features * n_units_per_cluster, n_clusters_estimation,
        1.0 / n_clusters_estimation)

    data = np.zeros((n_units_per_cluster * n_clusters_simulation, n_features, n_time_steps))

    return (estimated_parameters, true_parameters, true_states,
            predicted_cluster_probabilities, filtered_cluster_probabilities, data)
