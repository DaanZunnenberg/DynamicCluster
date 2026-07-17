"""Maximum-likelihood estimation of the GAS-HMM dynamic clustering model."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import loggamma

from .utils import vech, unvech, logit, logistic, flatten_time_major
from .simulation import compute_cluster_distance_matrix, map_distances_to_transition_probabilities

# Keys expected in the `estimation_params` dict passed to `estimate_maximum_likelihood`.
N_CLUSTERS = "n_clusters"
N_STARTING_VALUES = "n_starting_values"
VERBOSITY = "verbosity"
INVERSE_DEGREES_OF_FREEDOM = "inverse_degrees_of_freedom"
SMOOTHING = "smoothing"
MEAN_SPECIFIC_SMOOTHING = "mean_specific_smoothing"
WARM_START = "warm_start"
REGULARIZE_COVARIANCE = "regularize_covariance"
BURN_IN = "burn_in"


def restrict_parameters(state, parameter_vector, mean_specific_smoothing):
    """Apply parameter restrictions to `parameter_vector`.

    Currently a no-op kept for didactical purposes; no transformation is
    specified.

    Parameters
    ----------
    state : SimulationState
    parameter_vector : ndarray
    mean_specific_smoothing : int
        Flag for a mean-specific smoothing parameter.

    Returns
    -------
    ndarray
        The (possibly restricted) parameter vector.
    """
    return parameter_vector


def extract_parameters(state, parameter_vector, mean_specific_smoothing,
                        inverse_degrees_of_freedom):
    """Extract and transform model parameters from the optimizer's raw vector.

    Parameters
    ----------
    state : SimulationState
    parameter_vector : ndarray
        Raw parameter vector as seen by the optimizer.
    mean_specific_smoothing : int
        Flag for a mean-specific smoothing parameter.
    inverse_degrees_of_freedom : float
        Inverse degrees of freedom, for Student-t mixtures.

    Returns
    -------
    smoothing_params : ndarray, shape (n_features + 1,)
        `n_features` smoothing parameters for the means, plus one smoothing
        parameter for the covariance.
    gamma : float
        Decay parameter for the transition probabilities.
    degrees_of_freedom, inverse_degrees_of_freedom : float
        Degrees of freedom for Student-t mixtures.
    """
    ix = 0

    parameter_vector = restrict_parameters(state, parameter_vector, mean_specific_smoothing)

    n_features = 2

    # First element(s): smoothing parameter(s) for the mean.
    if mean_specific_smoothing == 1:
        smoothing_params = parameter_vector[ix:ix + n_features - 1 + 1]
        ix += n_features
    else:
        smoothing_params = np.ones(n_features) * parameter_vector[ix:ix + 1]
        ix += 1

    # No element for the covariance matrix's smoothing parameter.
    smoothing_params = np.append(smoothing_params, 0)

    # Next element: degrees of freedom.
    if np.isnan(inverse_degrees_of_freedom) or inverse_degrees_of_freedom == 0:
        inverse_degrees_of_freedom = np.nan
        degrees_of_freedom = np.nan

    elif inverse_degrees_of_freedom > 0:
        inverse_degrees_of_freedom = 0.5 * logistic(parameter_vector[ix])
        ix += 1
        degrees_of_freedom = 1.0 / inverse_degrees_of_freedom

    else:
        degrees_of_freedom = 1.0 / np.abs(inverse_degrees_of_freedom)
        ix += 1

    gamma = np.exp(parameter_vector[ix])
    ix += 1

    return smoothing_params, gamma, degrees_of_freedom, inverse_degrees_of_freedom


def run_kmeans_clustering(data, n_clusters, n_tries=10):
    """Run k-means clustering.

    Parameters
    ----------
    data : ndarray, shape (n_units, n_features)
        Data to be clustered.
    n_clusters : int
        Number of clusters.
    n_tries : int, optional
        Number of random restarts. Defaults to 10.

    Returns
    -------
    cluster_parameters : ndarray, shape (n_clusters, n_features * (n_features + 3) / 2)
        Each row holds a cluster's means, followed by the vech() of the
        Cholesky factor of its covariance matrix.
    assignments : ndarray, shape (n_units, n_clusters)
        One-hot cluster assignment: 1 if unit i belongs to cluster j, else 0.
    """
    n_units = data.shape[0]
    n_features = data.shape[1]
    best_mse = np.nan
    best_assignments = np.zeros(n_units)

    for _ in range(n_tries):
        # Initialize random cluster assignments.
        assignments = np.floor(np.random.uniform(
            0, 1, n_units) * n_clusters).astype(int)
        previous_assignments = np.zeros(len(assignments))
        assignments_changed = 1

        # Iterate until cluster assignment is stable/has converged.
        while assignments_changed != 0:
            # Compute the cluster means.
            one_hot_assignments = np.identity(n_clusters)[assignments, :]
            means = one_hot_assignments.T @ data / one_hot_assignments.sum(axis=0)[:, None]

            # Compute the mean squared error for this assignment.
            mse = ((data - means[assignments, :])**2).sum()

            # Update cluster assignments.
            squared_distances = np.zeros((n_units, n_clusters))
            for j in range(n_clusters):
                squared_distances[:, j] = ((data - means[j, :])**2).sum(axis=1)

            assignments = np.argmin(squared_distances, axis=1)

            assignments_changed = np.abs(
                assignments - previous_assignments).sum()
            previous_assignments = assignments

        # Keep this run if it improves on the current best.
        if np.isnan(best_mse) or mse < best_mse:
            best_mse = mse
            cluster_parameters = means
            best_assignments = assignments

    # Compute covariances.
    cluster_parameters = np.column_stack((cluster_parameters, np.zeros(
        (n_clusters, int((n_features * (n_features + 1)) / 2)))))

    for i in range(n_clusters):
        cluster_parameters[i, n_features:] = vech(np.linalg.cholesky(
            np.cov(data[best_assignments == i].T)))

    return cluster_parameters, np.identity(n_clusters)[best_assignments, :]


def initialize_parameter_vector(state, n_units_per_cluster, n_clusters, n_features,
                                 inverse_degrees_of_freedom,
                                 mean_specific_smoothing, warm_start):
    """Initialize the optimizer's starting parameter vector and cluster assignments.

    Parameters
    ----------
    state : SimulationState
    n_units_per_cluster : int
        Number of firms per mixture component.
    n_clusters : int
        Number of mixture components.
    n_features : int
        Number of characteristics.
    inverse_degrees_of_freedom : float
        Inverse degrees of freedom, for Student-t mixtures.
    mean_specific_smoothing : int
        Flag for a mean-specific smoothing parameter.
    warm_start : float
        Starting value for the smoothing parameter.

    Returns
    -------
    parameter_vector : ndarray
        Initial parameter vector for the optimizer.
    cluster_parameters : ndarray
        Cluster parameters from the k-means initialization.
    """
    # Mean smoothing parameter.
    if mean_specific_smoothing == 1:
        parameter_vector = np.ones(n_features) * warm_start
    else:
        parameter_vector = np.array([warm_start])

    # Use k-means cluster assignments as the starting predicted cluster probabilities.
    cluster_parameters, state.predicted_cluster_probabilities[0, :, :] = run_kmeans_clustering(
        state.data[:, :, 0], n_clusters)

    state.initial_predicted_probabilities = state.predicted_cluster_probabilities[0, :, :]

    # Degrees of freedom.
    if inverse_degrees_of_freedom > 0:
        transformed_dof = logit(2 * inverse_degrees_of_freedom)
        parameter_vector = np.hstack((parameter_vector, transformed_dof))

    # Transition probability decay parameter.
    parameter_vector = np.hstack((parameter_vector, np.log(3)))

    return parameter_vector, cluster_parameters


def mean_and_covariance_from_parameters(parameters, n_features, regularize_covariance=0):
    """Extract the mean vector and covariance matrix from a single cluster's parameter vector.

    Parameters
    ----------
    parameters : ndarray
        Cluster parameter vector: means followed by vech() of the covariance
        Cholesky factor.
    n_features : int
        Number of dimensions.
    regularize_covariance : float, optional
        Regularization term added to the covariance matrix's diagonal.
        Defaults to 0.

    Returns
    -------
    mean : ndarray
    covariance : ndarray
    cholesky_factor : ndarray
        Cholesky decomposition of the (regularized) covariance matrix.
    """
    ix = 0
    mean = parameters[int(ix):int(ix + n_features - 1 + 1)]
    ix += n_features
    cholesky_factor = np.tril(unvech(parameters[int(ix):int(ix + n_features * (n_features + 1) / 2 - 1) + 1]))
    covariance = cholesky_factor @ cholesky_factor.T + regularize_covariance * np.identity(n_features)
    ix += n_features * (n_features + 1) / 2
    cholesky_factor = np.linalg.cholesky(covariance)

    return mean, covariance, cholesky_factor


def run_hmm_gas_filter(state, smoothing_params, gamma, initial_predicted_cluster_probabilities,
                        smoothing, regularize_covariance, inverse_degrees_of_freedom):
    """Run the GAS filter for the cluster means and covariance matrices.

    The filter is currently a single-parameter EWMA-style score-driven update.

    Parameters
    ----------
    state : SimulationState
    smoothing_params : ndarray, shape (n_features + 1,)
        GAS smoothing parameter(s) for the mean (elements 0..n_features-1)
        and for the covariance (element n_features).
    gamma : float
        Decay rate of the transition probabilities between clusters, in
        terms of the distance between cluster means:
            Pi(l, m) = exp(-gamma * d(l, m)) / (1 + sum_(q != l) exp(-gamma * d(l, q)))
        where d(l, q) is the distance between the means of clusters l and q.
    initial_predicted_cluster_probabilities : ndarray
        Initial predicted cluster probabilities.
    smoothing : float
        Smoothing fraction used for estimation.
    regularize_covariance : float
        This value times the identity matrix is added to each covariance
        matrix to prevent degenerate results.
    inverse_degrees_of_freedom : float
        Inverse degrees of freedom, for Student-t mixtures.

    Returns
    -------
    log_likelihood : ndarray, shape (n_time_steps,)
        Log-likelihood contributions at each time point t = 1, ..., n_time_steps.
    """
    n_units = state.data.shape[0]
    n_features = state.data.shape[1]
    n_time_steps = state.data.shape[2]
    n_clusters = state.cluster_parameters.shape[0]

    log_lik_by_cluster = np.zeros((n_units, n_clusters))
    log_likelihood = np.zeros(n_time_steps)

    predicted_cluster_probabilities = initial_predicted_cluster_probabilities
    parameters = state.cluster_parameters.copy()

    for t in range(n_time_steps):
        # Store the time-varying mean/scale parameters and predicted probabilities.
        state.estimated_parameters[t, :, :] = parameters.copy()
        state.predicted_cluster_probabilities[t, :, :] = predicted_cluster_probabilities.copy()

        for j in range(n_clusters):
            mean, covariance, cholesky_factor = mean_and_covariance_from_parameters(
                parameters[j, :].copy(), n_features, regularize_covariance)
            inverse_covariance = np.linalg.inv(covariance)

            if np.isnan(mean).any():
                return np.array([-999999] * n_time_steps)

            # Log likelihood contribution of cluster j at time t.
            if np.isnan(inverse_degrees_of_freedom) or inverse_degrees_of_freedom == 0:
                # Gaussian case.
                log_lik_by_cluster[:, j] = (-0.5 * n_features * np.log(2 * np.pi)
                                  - np.log(np.diag(cholesky_factor)).sum()
                                  - (((state.data[:, :, t] - mean) @ inverse_covariance) *
                                    (state.data[:, :, t] - mean)).sum(axis=1)
                                  )
            else:
                # Student-t case.
                log_lik_by_cluster[:, j] = (loggamma(0.5 * (1 / inverse_degrees_of_freedom + n_features))
                                  - loggamma(0.5 / inverse_degrees_of_freedom)
                                  - 0.5 * n_features * np.log(2 * np.pi)
                                  - np.log(np.diag(cholesky_factor)).sum()
                                  - 0.5 * (1 / inverse_degrees_of_freedom + n_features) *
                                  np.log(1 + np.abs(1 / inverse_degrees_of_freedom) *
                                      ((state.data[:, :, t] - mean) @ inverse_covariance * (state.data[:, :, t] - mean)).sum(axis=1))
                                  )

        # Combine log-likelihoods across clusters for time t, using the
        # standard numerically-stable log-sum-exp trick.
        max_log_lik = np.max(log_lik_by_cluster, axis=1)

        filtered_cluster_probabilities = np.exp(log_lik_by_cluster - max_log_lik[:, None]) * predicted_cluster_probabilities

        # Sum across units, again subtracting the largest component before
        # exponentiation for numerical stability.
        log_likelihood[t] = (max_log_lik + np.log(filtered_cluster_probabilities.sum(axis=1))).sum()

        # Normalize to get the filtered probabilities, used in the GAS update below.
        filtered_cluster_probabilities = (filtered_cluster_probabilities /
                                         filtered_cluster_probabilities.sum(axis=1)[:, None])

        state.filtered_cluster_probabilities[t, :, :] = filtered_cluster_probabilities.copy()

        # Compute (smoothed) cluster distances.
        if t == 0:
            cluster_distances = compute_cluster_distance_matrix(parameters[:, :n_features].copy(),
                                                       parameters[:, n_features:].copy(),
                                                       gamma)
        else:
            cluster_distances = (1 - smoothing) * cluster_distances + \
                smoothing * compute_cluster_distance_matrix(parameters[:, :n_features].copy(),
                                                 parameters[:, n_features:].copy(),
                                                 gamma)

        # Transition probabilities from t to t+1, based on the means at time t.
        transition_probabilities = map_distances_to_transition_probabilities(cluster_distances, gamma)

        # Update the cluster means and covariances via the GAS score step.
        for j in range(n_clusters):
            mean, covariance, cholesky_factor = mean_and_covariance_from_parameters(
                parameters[j, :].copy(), n_features, regularize_covariance)
            inverse_covariance = np.linalg.inv(covariance)

            # Mean's score step.
            mean_score = (filtered_cluster_probabilities[:, j].T @ (state.data[:, :, t] - mean)
                      / filtered_cluster_probabilities[:, j].sum()
                      )
            mean += smoothing_params[:n_features] * mean_score

            # Covariance matrix's score step.
            covariance_score = ((state.data[:, :, t] - mean).T
                      @ (filtered_cluster_probabilities[:, j, None] * (state.data[:, :, t] - mean))
                      / filtered_cluster_probabilities[:, j].sum()
                      - covariance
                      )

            covariance += smoothing_params[n_features] * covariance_score
            if np.isnan(parameters).any():
                return np.array([-999999] * n_time_steps)

            # Store the updated parameters.
            ix = 0
            parameters[j, int(ix):int(ix + n_features - 1 + 1)] = mean.copy()

            ix += n_features
            parameters[j, int(ix):int(ix + n_features * (n_features + 1) / 2)] = vech(cholesky_factor).copy()
            ix += n_features * (n_features + 1) / 2

        # Update predicted probabilities for the next period.
        predicted_cluster_probabilities = filtered_cluster_probabilities @ transition_probabilities

    return log_likelihood


def estimate_maximum_likelihood(state, estimation_params):
    """Estimate the GAS-HMM model's parameters by maximum likelihood.

    Parameters
    ----------
    state : SimulationState
    estimation_params : dict
        Estimation configuration, keyed by the module-level constants
        `N_CLUSTERS`, `N_STARTING_VALUES`, etc.

    Returns
    -------
    ndarray
        Stacked estimation results: optimizer output, restricted parameters,
        estimated cluster-specific parameters, filtered cluster
        probabilities, true parameters, true states, and simulated data.
    """
    n_clusters = estimation_params[N_CLUSTERS]
    n_starting_values = estimation_params[N_STARTING_VALUES]
    inverse_degrees_of_freedom = estimation_params[INVERSE_DEGREES_OF_FREEDOM]
    smoothing = estimation_params[SMOOTHING]
    mean_specific_smoothing = estimation_params[MEAN_SPECIFIC_SMOOTHING]
    warm_start = estimation_params[WARM_START]
    regularize_covariance = estimation_params[REGULARIZE_COVARIANCE]
    burn_in = estimation_params[BURN_IN]

    n_features = state.data.shape[1]
    n_units_per_cluster = state.data.shape[0]
    n_time_steps = state.data.shape[2]

    results = None

    for start_index in range(n_starting_values):
        parameter_vector, state.cluster_parameters = initialize_parameter_vector(
            state,
            n_units_per_cluster,
            n_clusters,
            n_features,
            inverse_degrees_of_freedom,
            mean_specific_smoothing,
            warm_start)

        def negative_average_log_likelihood(vP):
            if np.abs(vP).max(axis=0) > 50:
                return 900000000

            smoothing_params, gamma = extract_parameters(state,
                                          vP,
                                            mean_specific_smoothing,
                                            inverse_degrees_of_freedom)[0:2]

            if np.abs(np.log(gamma)) > 5:
                return 900000000
            if np.isnan(state.initial_predicted_probabilities).any():
                return 900000000

            log_likelihood = run_hmm_gas_filter(state,
                                smoothing_params,
                                gamma,
                                state.initial_predicted_probabilities.copy(),
                                smoothing,
                                regularize_covariance,
                                inverse_degrees_of_freedom)

            if np.isnan(log_likelihood[burn_in:].mean()):
                return 900000000
            else:
                return -log_likelihood[burn_in:].mean()

        minimize(negative_average_log_likelihood,
              x0=parameter_vector,
              method='BFGS',
              options={'maxiter': 20, 'gtol': 1e-4})

        # Use the t=1 filtered probabilities as the initial predicted probabilities.
        state.initial_predicted_probabilities = state.filtered_cluster_probabilities[0, :, :].copy()

        result = minimize(negative_average_log_likelihood,
              x0=parameter_vector,
              method='BFGS',
              options={'maxiter': 100, 'gtol': 1e-4})

        print(f'\n BFGS exit status {result.status} and succes:{result.success}.\n{result.message}\n')
        print(result.x)
        print(result)

        # Switch regimes around if observation 1 does not have the highest
        # probability. Note the time paths themselves are unchanged; the
        # filter must be rerun for that.
        if (n_clusters == 2) and (state.filtered_cluster_probabilities[0, 0, 0] < state.filtered_cluster_probabilities[0, 0, 1]):
            for t in range(n_time_steps):
                state.filtered_cluster_probabilities[t, :, :] = (
                (state.filtered_cluster_probabilities[t, :, :].copy()
                 @ np.array([[0, 1],
                             [1, 0]]))
                )
                state.estimated_parameters[t, :, :] = (np.array([[0, 1], [1, 0]])
                                                 @ state.estimated_parameters[t, :, :].copy()
                                                 )

        # Store the results for this starting value.
        result_row = np.hstack((result.fun,
                                result.status,
                                result.x,
                                restrict_parameters(state, result.x, mean_specific_smoothing),
                                flatten_time_major(state.estimated_parameters),
                                flatten_time_major(state.filtered_cluster_probabilities)
                                ))
        if start_index == 0:
            results = [result_row]
        else:
            results = [results] + [result_row]

    results = np.vstack(results)
    best_result = results[0, :]

    return np.hstack((best_result,
                      flatten_time_major(state.true_parameters),
                      state.true_states.reshape(-1, 1).flatten(),
                      flatten_time_major(state.data)
                      ))
