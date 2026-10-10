import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pykalman.sqrt import (
    AdditiveUnscentedKalmanFilter,
    BiermanKalmanFilter,
    CholeskyKalmanFilter,
)


@pytest.mark.parametrize(
    "filter_class",
    [CholeskyKalmanFilter, BiermanKalmanFilter, AdditiveUnscentedKalmanFilter],
)
@pytest.mark.parametrize("n_dim_state", [1, 2])
@pytest.mark.parametrize("missing", [False, True])
def test_filter_update_returns_reusable_covariance(filter_class, n_dim_state, missing):
    transition = np.eye(n_dim_state) * 0.9
    observation = np.ones((1, n_dim_state))
    kwargs = dict(
        transition_covariance=np.eye(n_dim_state) * 0.1,
        observation_covariance=np.array([[0.5]]),
        initial_state_mean=np.zeros(n_dim_state),
        initial_state_covariance=np.eye(n_dim_state),
    )
    if filter_class is AdditiveUnscentedKalmanFilter:
        kwargs.update(
            transition_functions=lambda x: transition @ x,
            observation_functions=lambda x: observation @ x,
        )
    else:
        kwargs.update(transition_matrices=transition, observation_matrices=observation)
    estimator = filter_class(**kwargs)
    observations = np.ma.array([[0.1], [0.2], [-0.4], [0.8], [0.3]])
    if missing:
        observations[2] = np.ma.masked
    batch_means, batch_covariances = estimator.filter(observations)
    mean, covariance = batch_means[0], batch_covariances[0]
    for t in range(1, len(observations)):
        old_mean, old_covariance = mean.copy(), covariance.copy()
        new_mean, new_covariance = estimator.filter_update(
            mean, covariance, None if missing and t == 2 else observations[t]
        )
        assert_array_equal(mean, old_mean)
        assert_array_equal(covariance, old_covariance)
        assert new_mean.shape == (n_dim_state,)
        assert new_covariance.shape == (n_dim_state, n_dim_state)
        assert_allclose(new_mean, batch_means[t], atol=1e-12)
        assert_allclose(new_covariance, batch_covariances[t], atol=1e-12)
        mean, covariance = new_mean, new_covariance
