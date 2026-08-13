"""Tests for dynamiccluster.initialization: array allocation helpers."""

import numpy as np

from dynamiccluster.initialization import (
    initialize_time_varying_parameter_structure,
    initialize_simulation_matrices,
)


def test_initialize_time_varying_parameter_structure_shape_and_fill():
    array = initialize_time_varying_parameter_structure(3, 2, 4, fill_value=7)
    assert array.shape == (3, 2, 4)
    assert np.all(array == 7)


def test_initialize_time_varying_parameter_structure_casts_float_dims():
    array = initialize_time_varying_parameter_structure(3.0, 2.0, 4.0)
    assert array.shape == (3, 2, 4)


def test_initialize_simulation_matrices_shapes():
    n_time_steps, n_units_per_cluster, n_features = 5, 10, 2
    n_clusters_estimation, n_clusters_simulation = 2, 2

    (estimated_parameters, true_parameters, true_states,
     predicted_cluster_probabilities, filtered_cluster_probabilities,
     data) = initialize_simulation_matrices(
        n_time_steps, n_units_per_cluster, n_features,
        n_clusters_estimation, n_clusters_simulation)

    n_params_per_cluster = 0.5 * n_features * (n_features + 3)

    assert estimated_parameters.shape == (n_time_steps, n_clusters_estimation, n_params_per_cluster)
    assert true_parameters.shape == (n_time_steps, n_clusters_simulation, n_params_per_cluster)
    assert true_states.shape == (n_time_steps, n_clusters_simulation * n_units_per_cluster)
    assert data.shape == (n_units_per_cluster * n_clusters_simulation, n_features, n_time_steps)
    # Predicted/filtered probabilities start out uniform across clusters.
    np.testing.assert_allclose(predicted_cluster_probabilities[0, 0, :],
                                np.ones(n_clusters_estimation) / n_clusters_estimation)
    np.testing.assert_allclose(filtered_cluster_probabilities, predicted_cluster_probabilities)
