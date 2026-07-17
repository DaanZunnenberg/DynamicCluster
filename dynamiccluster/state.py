"""State container shared across simulation and estimation."""


class SimulationState:
    """Holds simulated data, true/estimated parameters, and cluster
    probabilities for one simulation-estimation cycle.

    Attributes
    ----------
    data : ndarray, shape (n_units, n_features, n_time_steps)
        Simulated panel data.
    estimated_parameters : ndarray, shape (n_time_steps, n_clusters, n_params)
        Time-varying cluster parameters (means + covariance Cholesky factors)
        produced by estimation.
    true_parameters : ndarray, shape (n_time_steps, n_clusters, n_params)
        Time-varying cluster parameters used to generate the data.
    true_states : ndarray, shape (n_time_steps, n_units)
        The cluster each unit truly belongs to at each time step.
    predicted_cluster_probabilities : ndarray
        One-step-ahead predicted cluster membership probabilities per unit.
    filtered_cluster_probabilities : ndarray
        Filtered (in-sample) cluster membership probabilities per unit.
    initial_predicted_probabilities : ndarray
        Predicted cluster probabilities used to start the GAS/HMM filter.
    cluster_parameters : ndarray
        Current cluster parameters used while running the likelihood filter.
    """

    def __init__(self):
        self.data = []
        self.estimated_parameters = []
        self.true_parameters = []
        self.true_states = []
        self.predicted_cluster_probabilities = []
        self.filtered_cluster_probabilities = []
        self.initial_predicted_probabilities = []
        self.cluster_parameters = []
