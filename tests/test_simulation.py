"""Tests for dynamiccluster.simulation: the data-generating process."""

import numpy as np
import pytest

from dynamiccluster.state import SimulationState
from dynamiccluster.initialization import initialize_simulation_matrices
from dynamiccluster.simulation import (
    simulate_data,
    map_distances_to_transition_probabilities,
    compute_cluster_distance_matrix,
    N_UNITS_PER_CLUSTER, N_FEATURES, N_CLUSTERS, INVERSE_DEGREES_OF_FREEDOM,
    N_TIME_STEPS, CORRELATION_MATRIX, CIRCLE_MIDPOINT_DISTANCE, CIRCLE_RADIUS,
    N_LAPS, SMOOTHING, TIME_VARYING_PROBABILITIES, GAMMA, SEMI_MAJOR_AXIS,
    ELLIPSE_1_ORIENTATION, ELLIPSE_2_ORIENTATION, SIMULATION_TYPE,
)
from dynamiccluster.utils import vech


def _small_simulation_params(simulation_type=3, n_time_steps=6, n_units_per_cluster=5):
    correlation_matrix = np.array([[1.0, -0.5], [-0.5, 1.0]])
    return {
        N_UNITS_PER_CLUSTER: n_units_per_cluster,
        N_FEATURES: 2,
        N_CLUSTERS: 2,
        INVERSE_DEGREES_OF_FREEDOM: 0,
        N_TIME_STEPS: n_time_steps,
        CORRELATION_MATRIX: correlation_matrix,
        CIRCLE_MIDPOINT_DISTANCE: 2,
        CIRCLE_RADIUS: 2,
        N_LAPS: 1,
        SMOOTHING: 0.1,
        TIME_VARYING_PROBABILITIES: 1,
        GAMMA: 0.25,
        SEMI_MAJOR_AXIS: 2,
        ELLIPSE_1_ORIENTATION: 1,
        ELLIPSE_2_ORIENTATION: 0,
        SIMULATION_TYPE: simulation_type,
    }


def test_map_distances_to_transition_probabilities_rows_sum_to_one():
    distance_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    transition_matrix = map_distances_to_transition_probabilities(distance_matrix, gamma=0.5)
    np.testing.assert_allclose(transition_matrix.sum(axis=1), np.ones(2))
    assert np.all(transition_matrix >= 0)


def test_compute_cluster_distance_matrix_symmetric_and_zero_diagonal():
    cluster_means = np.array([[0.0, 0.0], [3.0, 4.0]])
    cholesky_factors = np.tile(vech(np.eye(2)), (2, 1))
    distance_matrix = compute_cluster_distance_matrix(cluster_means, cholesky_factors, gamma=0.25)
    assert distance_matrix.shape == (2, 2)
    np.testing.assert_allclose(np.diag(distance_matrix), [0.0, 0.0])
    assert distance_matrix[0, 1] == pytest.approx(distance_matrix[1, 0])
    assert distance_matrix[0, 1] == pytest.approx(5.0)


@pytest.mark.parametrize("simulation_type", [0, 1, 2, 3])
def test_simulate_data_produces_finite_data_for_all_simulation_types(simulation_type):
    np.random.seed(0)
    n_time_steps, n_units_per_cluster, n_features = 6, 5, 2
    n_clusters = 2

    state = SimulationState()
    (state.estimated_parameters, state.true_parameters, state.true_states,
     state.predicted_cluster_probabilities, state.filtered_cluster_probabilities,
     state.data) = initialize_simulation_matrices(
        n_time_steps, n_units_per_cluster, n_features, n_clusters, n_clusters)

    simulation_params = _small_simulation_params(simulation_type, n_time_steps, n_units_per_cluster)
    simulate_data(state, simulation_params)

    assert state.data.shape == (n_units_per_cluster * n_clusters, n_features, n_time_steps)
    assert np.isfinite(state.data).all()
    # Every unit should be assigned to a valid cluster index at every time step.
    assert np.all((state.true_states >= 0) & (state.true_states < n_clusters))


def test_simulate_data_is_reproducible_given_seed():
    n_time_steps, n_units_per_cluster, n_features, n_clusters = 5, 4, 2, 2

    def run():
        np.random.seed(42)
        state = SimulationState()
        (state.estimated_parameters, state.true_parameters, state.true_states,
         state.predicted_cluster_probabilities, state.filtered_cluster_probabilities,
         state.data) = initialize_simulation_matrices(
            n_time_steps, n_units_per_cluster, n_features, n_clusters, n_clusters)
        simulate_data(state, _small_simulation_params(3, n_time_steps, n_units_per_cluster))
        return state.data

    data_1 = run()
    data_2 = run()
    np.testing.assert_allclose(data_1, data_2)
