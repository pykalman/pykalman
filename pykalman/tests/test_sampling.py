"""Regression tests for sampling from numerically equivalent models."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pykalman import AdditiveUnscentedKalmanFilter, KalmanFilter, UnscentedKalmanFilter
from pykalman.sqrt import (
    AdditiveUnscentedKalmanFilter as SqrtAdditiveUnscentedKalmanFilter,
)
from pykalman.sqrt import BiermanKalmanFilter, CholeskyKalmanFilter


@pytest.mark.parametrize(
    "kf_cls",
    [
        KalmanFilter,
        BiermanKalmanFilter,
        CholeskyKalmanFilter,
        UnscentedKalmanFilter,
        AdditiveUnscentedKalmanFilter,
        SqrtAdditiveUnscentedKalmanFilter,
    ],
)
@pytest.mark.parametrize(
    "covariance_name", ["transition_covariance", "observation_covariance"]
)
@pytest.mark.parametrize("covariance_dtype", [np.float32, np.float64])
def test_sample_non_native_covariance(kf_cls, covariance_name, covariance_dtype):
    covariance = np.array([[1.0, 0.2], [0.2, 2.0]], dtype=covariance_dtype)
    parameters = {
        "transition_covariance": covariance,
        "observation_covariance": covariance,
    }
    states, observations = kf_cls(**parameters).sample(
        5, initial_state=[0.0, 0.0], random_state=7
    )

    parameters[covariance_name] = covariance.astype(covariance.dtype.newbyteorder("S"))
    swapped_states, swapped_observations = kf_cls(**parameters).sample(
        5, initial_state=[0.0, 0.0], random_state=7
    )

    assert_array_equal(swapped_states, states)
    assert_array_equal(swapped_observations, observations)
