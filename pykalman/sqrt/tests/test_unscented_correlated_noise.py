"""Regression tests for correlated additive noise in the square-root UKF."""

import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_allclose

from pykalman.sqrt import AdditiveUnscentedKalmanFilter


@pytest.mark.parametrize("correlation", [-0.1, 0.0, 0.1])
def test_missing_observations_preserve_process_covariance(correlation):
    # For a random walk with no measurements, Var(x_t) = P_0 + t * Q.
    initial_mean = np.array([0.3, -0.4])
    initial_covariance = np.array([[1.0, 0.25], [0.25, 0.8]])
    process_covariance = np.array([[0.3, correlation], [correlation, 0.2]])
    original_covariance = process_covariance.copy()
    observations = ma.masked_all((4, 1))
    kf = AdditiveUnscentedKalmanFilter(
        transition_functions=lambda x: x,
        observation_functions=lambda x: x[:1],
        transition_covariance=process_covariance,
        observation_covariance=np.array([[0.6]]),
        initial_state_mean=initial_mean,
        initial_state_covariance=initial_covariance,
    )

    means, covariances = kf.filter(observations)

    assert_allclose(means, np.tile(initial_mean, (4, 1)), atol=1e-12)
    expected = initial_covariance + np.arange(4)[:, None, None] * process_covariance
    assert_allclose(covariances, expected, atol=1e-12)
    assert_allclose(process_covariance, original_covariance, atol=0)


@pytest.mark.parametrize("method", ["filter", "smooth"])
@pytest.mark.parametrize("n_dim_obs", [1, 2])
def test_correlated_random_walk_matches_joint_gaussian(method, n_dim_obs):
    initial_mean = np.array([0.3, -0.4])
    initial_covariance = np.array([[1.0, 0.25], [0.25, 0.8]])
    process_covariance = np.array([[0.3, 0.1], [0.1, 0.2]])
    observation_matrix = np.array([[1.0, 0.3], [-0.2, 1.0]])[:n_dim_obs]
    noise_covariance = np.array([[0.6, 0.15], [0.15, 0.4]])[:n_dim_obs, :n_dim_obs]
    observations = ma.array([[0.2, -0.5], [0.0, 0.0], [-0.1, 0.5], [0.4, 0.2]])
    observations = observations[:, :n_dim_obs]
    observations[1] = ma.masked
    kf = AdditiveUnscentedKalmanFilter(
        transition_functions=lambda x: x,
        observation_functions=lambda x: observation_matrix @ x,
        transition_covariance=process_covariance,
        observation_covariance=noise_covariance,
        initial_state_mean=initial_mean,
        initial_state_covariance=initial_covariance,
    )

    # Joint random-walk covariance: Cov(x_t, x_s) = P_0 + min(t, s) * Q.
    # Condition this Gaussian directly, independently of the filtering recurrence.
    times = np.arange(4)
    state_covariance = np.kron(np.ones((4, 4)), initial_covariance) + np.kron(
        np.minimum.outer(times, times), process_covariance
    )
    observation_map = np.kron(np.eye(4), observation_matrix)
    cross_covariance = state_covariance @ observation_map.T
    observation_covariance = observation_map @ cross_covariance + np.kron(
        np.eye(4), noise_covariance
    )
    observation_mean = np.tile(observation_matrix @ initial_mean, 4)
    expected_means, expected_covariances = [], []
    for t in times:
        used = ~ma.getmaskarray(observations).any(axis=1)
        if method == "filter":
            used[t + 1 :] = False
        used = np.repeat(used, n_dim_obs)
        state = slice(2 * t, 2 * t + 2)
        cross = cross_covariance[state][:, used]
        gain = np.linalg.solve(observation_covariance[np.ix_(used, used)], cross.T).T
        expected_means.append(
            initial_mean
            + gain @ (observations.data.ravel()[used] - observation_mean[used])
        )
        expected_covariances.append(
            initial_covariance + t * process_covariance - gain @ cross.T
        )

    means, covariances = getattr(kf, method)(observations)

    assert_allclose(means, expected_means, atol=1e-12)
    assert_allclose(covariances, expected_covariances, atol=1e-12)
