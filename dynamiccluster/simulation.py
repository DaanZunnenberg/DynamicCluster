"""Data-generating process for the dynamic clustering simulation study."""

import numpy as np

from .utils import vech, unvech

# Keys expected in the `simulation_params` dict passed to `simulate_data`.
N_UNITS_PER_CLUSTER = "n_units_per_cluster"
N_FEATURES = "n_features"
N_CLUSTERS = "n_clusters"
INVERSE_DEGREES_OF_FREEDOM = "inverse_degrees_of_freedom"
N_TIME_STEPS = "n_time_steps"
CORRELATION_MATRIX = "correlation_matrix"
CIRCLE_MIDPOINT_DISTANCE = "circle_midpoint_distance"
CIRCLE_RADIUS = "circle_radius"
N_LAPS = "n_laps"
SMOOTHING = "smoothing"
TIME_VARYING_PROBABILITIES = "time_varying_probabilities"
GAMMA = "gamma"
SEMI_MAJOR_AXIS = "semi_major_axis"
ELLIPSE_1_ORIENTATION = "ellipse1_orientation"
ELLIPSE_2_ORIENTATION = "ellipse2_orientation"
SIMULATION_TYPE = "simulation_type"


def compute_initial_cluster_means(
        simulation_type,
        time_ahead,
        n_time_steps,
        cluster_centers,
        circle_radius,
        time_vector,
        circle_midpoint_distance,
        time_varying_probabilities,
        semi_major_axis,
        ellipse1_orientation,
        ellipse2_orientation):
    """Compute the initial cluster means for the chosen simulation type.

    Parameters
    ----------
    simulation_type : int
        0 for ovals, 1 for converging circles, 2 for Y-shaped, 3 for flipped-Y.
    time_ahead : int
        How many time steps the second cluster leads the first by.
    n_time_steps : int
        Total number of time steps in the simulation.
    cluster_centers : ndarray, shape (2, 2)
        Center of each cluster's trajectory.
    circle_radius : float
        Radius of the circular/elliptical trajectories.
    time_vector : ndarray
        Time steps of the simulation, in radians.
    circle_midpoint_distance : float
        Distance between the centers of the two clusters.
    time_varying_probabilities : int
        1 if transition probabilities (and thus true-mean trajectories) are
        time-varying, 0 if they are constant.
    semi_major_axis : float
        Semi-major axis of the elliptical trajectory.
    ellipse1_orientation, ellipse2_orientation : int
        Orientation of the first and second cluster's ellipse.

    Returns
    -------
    initial_means : ndarray, shape (2, 2)
        Initial means of the simulation.
    cluster_centers : ndarray, shape (2, 2)
        Center of each cluster's trajectory (may be recomputed for some
        simulation types).
    """
    if simulation_type == 1:
        # Converging circles: fix angles to half a rotation.
        time_vector = 2 * np.pi * np.arange(1, n_time_steps + 1) / (2 * n_time_steps)

        cluster_centers = np.array([[-circle_radius, 0.],
                                    [circle_radius, 0.]]
                                   )
        initial_means = cluster_centers + circle_radius * np.array([[0, 1],
                                                                     [0, 1]]
                                                                    )
    elif simulation_type == 2:
        # Y-shaped: circle_radius becomes the Y distance, circle_midpoint_distance
        # becomes the X distance. Clusters move along straight lines.
        cluster_centers = np.array([[circle_midpoint_distance, circle_radius],
                                    [-circle_midpoint_distance, circle_radius]]
                                   )
        initial_means = cluster_centers

    elif simulation_type == 3:
        # Flipped-Y: same as Y-shaped, but clusters start together and separate
        # after t = n_time_steps / 2.
        cluster_centers = np.array([[0., circle_radius],
                                    [0., circle_radius]]
                                   )
        initial_means = cluster_centers

    elif time_varying_probabilities == 1:
        # Elliptical trajectories.
        initial_means = cluster_centers + circle_radius * \
            np.array([[-(semi_major_axis**(1 - ellipse1_orientation)) * np.cos(0),
                        (semi_major_axis**(ellipse1_orientation)) * np.sin(0)],
                      [(semi_major_axis**(1 - ellipse2_orientation)) * np.cos(time_vector[time_ahead]),
                        (semi_major_axis**(ellipse2_orientation)) * np.sin(time_vector[time_ahead])]]
                      )

    else:
        # Synchronized elliptical trajectories.
        initial_means = cluster_centers + circle_radius * \
            np.array([[-(semi_major_axis**(1 - ellipse1_orientation)) * np.cos(0),
                       (semi_major_axis**(ellipse1_orientation)) * np.sin(0)],
                      [-(semi_major_axis**(1 - ellipse2_orientation)) * np.cos(time_vector[time_ahead]),
                       (semi_major_axis**(ellipse2_orientation)) * np.sin(time_vector[time_ahead])]]
                     )

    return initial_means, cluster_centers


def compute_transition_probabilities(
        t,
        cluster_distances,
        previous_means,
        cluster_cholesky_factors,
        gamma,
        smoothing):
    """Compute the state transition probability matrix at time `t`.

    Smooths the cluster distance matrix over time and maps the (smoothed)
    distances into transition probabilities.
    """
    if t == 0:
        cluster_distances = compute_cluster_distance_matrix(
            previous_means, cluster_cholesky_factors, gamma)
    else:
        cluster_distances = (1 - smoothing) * cluster_distances \
            + smoothing * compute_cluster_distance_matrix(previous_means,
                                                          cluster_cholesky_factors,
                                                          gamma)
    # Transition probabilities from t to t+1, based on the means at time t.
    transition_probabilities = map_distances_to_transition_probabilities(
        cluster_distances, gamma)

    return transition_probabilities, cluster_distances


def compute_cluster_distance_matrix(cluster_means, cluster_cholesky_factors, gamma):
    """Compute the Mahalanobis distance matrix between all clusters.

    Parameters
    ----------
    cluster_means : ndarray
        Mean of each cluster.
    cluster_cholesky_factors : ndarray
        Cholesky decomposition of the covariance matrix of each cluster.
    gamma : float
        Transition probability decay parameter (unused directly here, kept
        for a consistent call signature with the caller).

    Returns
    -------
    distance_matrix : ndarray
        Mahalanobis distance between all clusters.
    """
    n_clusters = cluster_means.shape[0]

    distance_matrix = np.zeros((n_clusters, n_clusters))
    cholesky_factor = np.tril(unvech(cluster_cholesky_factors.mean(axis=0)))

    inverse_cholesky_factor = np.linalg.inv(cholesky_factor).T

    for i in range(n_clusters - 1):
        for j in range(i, n_clusters):
            distance_matrix[i, j] = np.sqrt(
                (((cluster_means[i, :] - cluster_means[j, :]) @ inverse_cholesky_factor.T)**2).sum())
            distance_matrix[j, i] = distance_matrix[i, j]

    return distance_matrix


def map_distances_to_transition_probabilities(distance_matrix, gamma):
    """Map cluster mean distances into a transition probability matrix.

    The underlying model is
        Pi(l, m) = exp(-gamma * d(l, m)) / (1 + sum_(q != l) exp(-gamma * d(l, q)))
    where d(l, q) is the distance between the means of clusters l and q.

    Parameters
    ----------
    distance_matrix : ndarray, shape (n_clusters, n_clusters)
        Pairwise distances between cluster means.
    gamma : float
        Transition probability decay parameter.

    Returns
    -------
    transition_matrix : ndarray, shape (n_clusters, n_clusters)
        Transition probabilities.
    """
    transition_matrix = np.exp(-gamma * distance_matrix)
    transition_matrix = transition_matrix / \
        transition_matrix.sum(axis=1, keepdims=True)

    return transition_matrix


def sample_new_unit_states(current_states, transition_probabilities):
    """Sample new cluster membership for each unit given the current states
    and the state transition probabilities.

    Parameters
    ----------
    current_states : ndarray
        Current cluster membership of each unit.
    transition_probabilities : ndarray
        State transition probability matrix.

    Returns
    -------
    new_states : ndarray
        New cluster membership of each unit.
    """
    cumulative_probabilities = transition_probabilities.cumsum(axis=1)[current_states]
    new_states = (cumulative_probabilities
                  < np.random.uniform(0, 1, len(current_states))[:, None]).sum(axis=1)
    return new_states


def update_cluster_means(previous_means,
                          simulation_type,
                          t,
                          cluster2_time_index,
                          n_time_steps,
                          cluster_centers,
                          circle_radius,
                          time_vector,
                          circle_midpoint_distance,
                          time_varying_probabilities,
                          semi_major_axis,
                          ellipse1_orientation,
                          ellipse2_orientation):
    """Update the cluster means for the current time step and simulation type.

    Parameters
    ----------
    previous_means : ndarray, shape (2, 2)
        Cluster means before the update.
    simulation_type : int
        0 for ovals, 1 for converging circles, 2 for Y-shaped, 3 for flipped-Y.
    t : int
        Current time step.
    cluster2_time_index : int
        Time index used to look up cluster 2's position (it is offset ahead
        of cluster 1 by a fixed number of steps).
    n_time_steps : int
        Total number of time steps.
    cluster_centers : ndarray, shape (2, 2)
        Center of each cluster's trajectory.
    circle_radius : float
        Radius of the circular/elliptical trajectories.
    time_vector : ndarray
        Time steps of the simulation, in radians.
    circle_midpoint_distance : float
        Distance between the centers of the two clusters.
    time_varying_probabilities : int
        1 if transition probabilities are time-varying, 0 if constant.
    semi_major_axis : float
        Semi-major axis of the elliptical trajectory.
    ellipse1_orientation, ellipse2_orientation : int
        Orientation of the first and second cluster's ellipse.

    Returns
    -------
    updated_means : ndarray, shape (2, 2)
    """
    if simulation_type == 1:
        if t < n_time_steps / 2:  # Otherwise the means stay put.
            updated_means = cluster_centers + circle_radius * \
                np.array([[np.sin(time_vector[t]), np.cos(time_vector[t])],
                          [-np.sin(time_vector[t]), np.cos(time_vector[t])]]
                         )
        else:
            updated_means = previous_means

    elif simulation_type == 2:
        # Take linear steps.
        current_x = ((t + 1 < n_time_steps / 2) * 2 * (t + 1) * circle_midpoint_distance /
                     n_time_steps) + ((t + 1 >= n_time_steps / 2) * circle_midpoint_distance)

        updated_means = cluster_centers + \
            np.array([[-current_x, -(t + 1) * 2 * circle_radius / n_time_steps],
                      [current_x, -(t + 1) * 2 * circle_radius / n_time_steps]]
                     )

    elif simulation_type == 3:
        # Take linear steps.
        current_x = ((t + 1 >= n_time_steps / 2)
                     * 2 * (t + 1 - n_time_steps / 2)
                     * circle_midpoint_distance
                     / n_time_steps)

        updated_means = cluster_centers + \
            np.array([[-current_x, -(t + 1) * 2 * circle_radius / n_time_steps],
                      [current_x, -(t + 1) * 2 * circle_radius / n_time_steps]]
                     )

    elif time_varying_probabilities == 1:
        updated_means = cluster_centers + circle_radius * \
            np.array([[-(semi_major_axis**(1 - ellipse1_orientation)) * np.cos(time_vector[t]),
                       (semi_major_axis**(ellipse1_orientation)) * np.sin(time_vector[t])],
                      [(semi_major_axis**(1 - ellipse2_orientation)) * np.cos(time_vector[cluster2_time_index]),
                       (semi_major_axis**(ellipse2_orientation)) * np.sin(time_vector[cluster2_time_index])]]
                     )

    else:
        updated_means = cluster_centers + circle_radius * \
            np.array([[-(semi_major_axis**(1 - ellipse1_orientation)) * np.cos(time_vector[t]),
                       (semi_major_axis**(ellipse1_orientation)) * np.sin(time_vector[t])],
                      [-(semi_major_axis**(1 - ellipse2_orientation)) * np.cos(time_vector[cluster2_time_index]),
                       (semi_major_axis**(ellipse2_orientation)) * np.sin(time_vector[cluster2_time_index])]]
                     )
    return updated_means


def simulate_data(state, simulation_params):
    """Simulate panel data under the dynamic-clustering data-generating process.

    Updates `state.true_states`, `state.true_parameters`, and `state.data`
    in place.

    Parameters
    ----------
    state : SimulationState
        Object whose `true_parameters`, `true_states`, and `data` arrays are
        filled in by this function.
    simulation_params : dict
        Simulation configuration, keyed by the module-level constants
        `N_UNITS_PER_CLUSTER`, `N_FEATURES`, etc.
    """
    n_units_per_cluster = simulation_params[N_UNITS_PER_CLUSTER]
    n_features = simulation_params[N_FEATURES]
    n_clusters = simulation_params[N_CLUSTERS]
    n_time_steps = simulation_params[N_TIME_STEPS]
    correlation_matrix = simulation_params[CORRELATION_MATRIX]
    circle_midpoint_distance = simulation_params[CIRCLE_MIDPOINT_DISTANCE]
    circle_radius = simulation_params[CIRCLE_RADIUS]
    n_laps = simulation_params[N_LAPS]
    smoothing = simulation_params[SMOOTHING]
    time_varying_probabilities = simulation_params[TIME_VARYING_PROBABILITIES]
    gamma = simulation_params[GAMMA]
    semi_major_axis = simulation_params[SEMI_MAJOR_AXIS]
    ellipse1_orientation = simulation_params[ELLIPSE_1_ORIENTATION]
    ellipse2_orientation = simulation_params[ELLIPSE_2_ORIENTATION]
    simulation_type = simulation_params[SIMULATION_TYPE]

    cluster_distances = None
    # How many time steps the second cluster's trajectory leads the first by.
    time_ahead = int(np.ceil(n_time_steps / 4))

    time_vector = n_laps * 2 * np.pi * np.arange(1, n_time_steps + time_ahead + 1) / n_time_steps
    current_states = np.kron(np.arange(0, n_clusters - 1 + 1), np.ones(n_units_per_cluster)).astype(int)

    cluster1_cholesky = np.linalg.cholesky(correlation_matrix).T

    # Change sign of off-diagonal elements for the second cluster's covariance.
    cluster2_cholesky = np.transpose(np.linalg.cholesky(
        (correlation_matrix * (2 * np.eye(n_features) - np.ones((n_features, n_features))))))

    # Initialize (previous) means.
    cluster_centers = np.array([[0., 0.],
                                [circle_midpoint_distance, 0.]]
                               )

    # Depending on the simulation type, generate the initial means.
    previous_means, cluster_centers = compute_initial_cluster_means(
        simulation_type,
        time_ahead,
        n_time_steps,
        cluster_centers,
        circle_radius,
        time_vector,
        circle_midpoint_distance,
        time_varying_probabilities,
        semi_major_axis,
        ellipse1_orientation,
        ellipse2_orientation
    )

    for t in range(n_time_steps):
        # To change the starting position of cluster 2, pretend it is ahead in time.
        cluster2_time_index = t + time_ahead
        state.true_parameters[t, :, :] = np.column_stack((previous_means,
                                                      np.row_stack((vech(cluster1_cholesky).T,
                                                                    vech(cluster2_cholesky).T))
                                                      ))
        # Store true states.
        state.true_states[t, :] = current_states

        # First, generate the unit observations.
        shocks = (current_states[:, None] * np.random.randn(n_units_per_cluster * n_clusters, n_features) @ cluster1_cholesky) + \
            ((1 - current_states)[:, None] * np.random.randn(n_units_per_cluster * n_clusters, n_features) @ cluster2_cholesky)

        # Store the data for this time step.
        state.data[:, :, t] = previous_means[current_states, :] + shocks

        # Second, generate the transition probability matrix.
        transition_probabilities, cluster_distances = \
            compute_transition_probabilities(t,
                                            cluster_distances,
                                            previous_means,
                                            vech(np.identity(2)) * np.ones(n_clusters)[:, None],
                                            gamma,
                                            smoothing
                                            )

        # Third, sample new unit states.
        current_states = sample_new_unit_states(current_states, transition_probabilities)

        # Fourth, update the means.
        previous_means = update_cluster_means(previous_means,
                                     simulation_type,
                                      t,
                                      cluster2_time_index,
                                      n_time_steps,
                                      cluster_centers,
                                      circle_radius,
                                      time_vector,
                                      circle_midpoint_distance,
                                      time_varying_probabilities,
                                      semi_major_axis,
                                      ellipse1_orientation,
                                      ellipse2_orientation
                                      )
    return
