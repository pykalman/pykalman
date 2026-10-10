import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pykalman import KalmanFilter
from pykalman.sqrt import BiermanKalmanFilter


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("varying", [False, True])
def test_observation_whitening_promotes_integer_matrices(dtype, varying):
    observations = np.array([[0.2, 0.5], [0.8, -0.3], [0.4, 0.6], [-0.1, 0.3]])
    matrices = np.repeat(np.eye(2, dtype=dtype)[None], len(observations), axis=0)
    if varying:
        matrices[2:] = [[1, 1], [0, 1]]
    original = matrices.copy()
    kwargs = dict(
        transition_matrices=np.eye(2),
        observation_matrices=matrices,
        transition_covariance=np.eye(2) * 0.1,
        observation_covariance=np.array([[2.0, 0.3], [0.3, 3.0]]),
    )
    expected_means, expected_covariances = KalmanFilter(**kwargs).filter(observations)
    means, covariances = BiermanKalmanFilter(**kwargs).filter(observations)
    assert_allclose(means, expected_means, atol=1e-7)
    assert_allclose(covariances, expected_covariances, atol=1e-7)
    assert_array_equal(matrices, original)
