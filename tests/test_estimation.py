"""Tests for dynamiccluster.estimation: parameter extraction and the ML pipeline."""

import numpy as np
import pytest

from dynamiccluster.state import SimulationState
from dynamiccluster.initialization import initialize_simulation_matrices
from dynamiccluster.simulation import (
    simulate_data,
    N_UNITS_PER_CLUSTER, N_FEATURES, N_CLUSTERS, INVERSE_DEGREES_OF_FREEDOM,
    N_TIME_STEPS, CORRELATION_MATRIX, CIRCLE_MIDPOINT_DISTANCE, CIRCLE_RADIUS,
    N_LAPS, SMOOTHING, TIME_VARYING_PROBABILITIES, GAMMA, SEMI_MAJOR_AXIS,
    ELLIPSE_1_ORIENTATION, ELLIPSE_2_ORIENTATION, SIMULATION_TYPE,
)
from dynamiccluster.estimation import (
    extract_parameters,
    mean_and_covariance_from_parameters,
    run_kmeans_clustering,
    estimate_maximum_likelihood,
    N_CLUSTERS as EST_N_CLUSTERS, N_STARTING_VALUES, VERBOSITY,
    INVERSE_DEGREES_OF_FREEDOM as EST_INVERSE_DEGREES_OF_FREEDOM,
    SMOOTHING as EST_SMOOTHING, MEAN_SPECIFIC_SMOOTHING, WARM_START,
    REGULARIZE_COVARIANCE, BURN_IN,
)
from dynamiccluster.utils import vech


def test_mean_and_covariance_from_parameters_roundtrip():
    n_features = 2
    true_covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    true_cholesky = np.linalg.cholesky(true_covariance)
    parameters = np.hstack(([1.0, -1.0], vech(true_cholesky)))

    mean, covariance, cholesky_factor = mean_and_covariance_from_parameters(parameters, n_features)

    np.testing.assert_allclose(mean, [1.0, -1.0])
    np.testing.assert_allclose(covariance, true_covariance, atol=1e-10)
    np.testing.assert_allclose(cholesky_factor @ cholesky_factor.T, true_covariance, atol=1e-10)


def test_extract_parameters_single_smoothing_and_gamma():
    # parameter_vector = [smoothing, log(gamma)] with dof disabled.
    parameter_vector = np.array([0.3, np.log(2.0)])
    smoothing_params, gamma, degrees_of_freedom, inverse_dof = extract_parameters(
        state=None, parameter_vector=parameter_vector,
        mean_specific_smoothing=0, inverse_degrees_of_freedom=0)

    np.testing.assert_allclose(smoothing_params, [0.3, 0.3, 0.0])
    assert gamma == pytest.approx(2.0)
    assert np.isnan(degrees_of_freedom)
    assert np.isnan(inverse_dof)


def test_run_kmeans_clustering_recovers_well_separated_clusters():
    np.random.seed(0)
    cluster_a = np.random.randn(50, 2) + np.array([10.0, 10.0])
    cluster_b = np.random.randn(50, 2) + np.array([-10.0, -10.0])
    data = np.vstack([cluster_a, cluster_b])

    cluster_parameters, assignments = run_kmeans_clustering(data, n_clusters=2, n_tries=3)

    assert cluster_parameters.shape[0] == 2
    assert assignments.shape == (100, 2)
    # Every point assigned to exactly one cluster.
    np.testing.assert_allclose(assignments.sum(axis=1), np.ones(100))
    # The two recovered means should be far apart (clusters correctly separated).
    means = cluster_parameters[:, :2]
    assert np.linalg.norm(means[0] - means[1]) > 10


def test_estimate_maximum_likelihood_runs_end_to_end_and_returns_finite_result():
    np.random.seed(7)
    n_time_steps, n_units_per_cluster, n_features, n_clusters = 5, 8, 2, 2

    state = SimulationState()
    (state.estimated_parameters, state.true_parameters, state.true_states,
     state.predicted_cluster_probabilities, state.filtered_cluster_probabilities,
     state.data) = initialize_simulation_matrices(
        n_time_steps, n_units_per_cluster, n_features, n_clusters, n_clusters)

    correlation_matrix = np.array([[1.0, -0.5], [-0.5, 1.0]])
    simulation_params = {
        N_UNITS_PER_CLUSTER: n_units_per_cluster, N_FEATURES: n_features,
        N_CLUSTERS: n_clusters, INVERSE_DEGREES_OF_FREEDOM: 0,
        N_TIME_STEPS: n_time_steps, CORRELATION_MATRIX: correlation_matrix,
        CIRCLE_MIDPOINT_DISTANCE: 2, CIRCLE_RADIUS: 2, N_LAPS: 1,
        SMOOTHING: 0.1, TIME_VARYING_PROBABILITIES: 1, GAMMA: 0.25,
        SEMI_MAJOR_AXIS: 2, ELLIPSE_1_ORIENTATION: 1, ELLIPSE_2_ORIENTATION: 0,
        SIMULATION_TYPE: 3,
    }
    simulate_data(state, simulation_params)

    estimation_params = {
        EST_N_CLUSTERS: n_clusters, N_STARTING_VALUES: 1, VERBOSITY: 0,
        EST_INVERSE_DEGREES_OF_FREEDOM: 0, EST_SMOOTHING: 0.25,
        MEAN_SPECIFIC_SMOOTHING: 0, WARM_START: 1.7,
        REGULARIZE_COVARIANCE: 0, BURN_IN: 0,
    }

    result = estimate_maximum_likelihood(state, estimation_params)

    assert result.ndim == 1
    assert np.isfinite(result).all()
