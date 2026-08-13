"""Tests for dynamiccluster.utils: vectorization and logit/logistic helpers."""

import numpy as np
import pytest

from dynamiccluster.utils import vech, unvech, logit, logistic, flatten_time_major


def test_vech_unvech_roundtrip():
    matrix = np.array([[4.0, 1.0, 2.0],
                        [1.0, 5.0, 3.0],
                        [2.0, 3.0, 6.0]])
    vectorized = vech(matrix)
    rebuilt = unvech(vectorized)
    np.testing.assert_allclose(rebuilt, matrix)


def test_vech_length():
    matrix = np.eye(4)
    assert vech(matrix).shape[0] == 4 * (4 + 1) / 2


def test_logit_logistic_are_inverses():
    probabilities = np.array([0.1, 0.5, 0.9])
    values = logit(probabilities)
    recovered = logistic(values)
    np.testing.assert_allclose(recovered, probabilities, atol=1e-8)


def test_logistic_scalar_and_array_consistent():
    scalar_result = logistic(np.array([0.0]))
    np.testing.assert_allclose(scalar_result, [0.5])


def test_flatten_time_major_shape():
    array_3d = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
    flattened = flatten_time_major(array_3d)
    assert flattened.shape == (2 * 3 * 4,)
    # First 12 entries should be the first time slice, flattened.
    np.testing.assert_allclose(flattened[:12], array_3d[0].reshape(-1))
