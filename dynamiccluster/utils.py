"""Generic numerical helpers that are not specific to the clustering method."""

import numpy as np


def flatten_time_major(array_3d):
    """Flatten a (time, rows, cols) array into a single row vector, time-major.

    Parameters
    ----------
    array_3d : ndarray, shape (n_time_steps, n_rows, n_cols)

    Returns
    -------
    ndarray, shape (n_time_steps * n_rows * n_cols,)
    """
    first_slice = array_3d[0, :, :].reshape(1, -1)
    flattened = first_slice.reshape(first_slice.shape[1])
    for t in range(1, array_3d.shape[0]):
        time_slice = array_3d[t, :, :].reshape(1, -1)
        flattened = np.hstack((flattened, time_slice.reshape(time_slice.shape[1],)))

    return flattened


def logit(probabilities):
    """Inverse logistic transform: maps (0, 1) back onto the real line."""
    try:
        return np.log(probabilities) - np.log(np.ones(probabilities.shape) - probabilities)
    except AttributeError:
        return np.log(probabilities) - np.log(1 - probabilities)


def logistic(values):
    """Logistic transform: maps the real line onto (0, 1)."""
    try:
        return np.choose(values > 0,
                         [np.ones(values.shape) / (np.ones(values.shape) + np.exp(-values)),
                          np.exp(values) / (np.ones(values.shape) + np.exp(values))])
    except AttributeError:
        return np.choose(values > 0,
                         [1 / (1 + np.exp(-values)),
                          np.exp(values) / (1 + np.exp(values))]
                         )


def vech(matrix):
    """Half-vectorize a matrix: stack its upper-triangular entries column-wise.

    Parameters
    ----------
    matrix : ndarray, shape (m, n)

    Returns
    -------
    ndarray, the vech() of `matrix`.
    """
    return matrix.T[np.triu_indices_from(matrix)]


def unvech(vector):
    """Inverse of vech(): rebuild a symmetric matrix from its half-vectorized form.

    Adapted from the statsmodels implementation.

    Parameters
    ----------
    vector : ndarray, shape (n * (n + 1) / 2,)

    Returns
    -------
    ndarray, shape (n, n), symmetric matrix with `vector` on and above the diagonal.
    """
    # Solve n*(n+1)/2 = len(vector) for n, correcting for floating point error.
    n_rows = 0.5 * (-1 + np.sqrt(1 + 8 * len(vector)))
    n_rows = int(np.round(n_rows))

    matrix = np.zeros((n_rows, n_rows))
    matrix[np.triu_indices(n_rows)] = vector
    matrix = matrix + matrix.T

    # The diagonal was added to itself above, so halve it back.
    matrix[np.diag_indices(n_rows)] /= 2

    return matrix
