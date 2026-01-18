"""Tests for loglikelihood related logic."""

import pykalman as pk
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
import numpy as np


def test_custom_parameter_inference():
    """Test for verifying that you may perform parameter inference using loglikelihood."""

    def fun(params, y_):
        phi = sigmoid(params[0])
        sigma = np.exp(params[1])

        f = pk.KalmanFilter(
            transition_matrices=phi,
            transition_covariance=sigma,
            observation_offsets=0.1**2.0,
        )

        return -f.loglikelihood(y_)

    np.random.seed(0)
    f = pk.KalmanFilter(
        transition_matrices=0.98,
        transition_covariance=0.05**2.0,
        observation_covariance=0.1**2.0,
    )

    _, y = f.sample(500)

    y[[10, 25, 100], 0] = np.nan
    x0 = np.array([0.5, np.log(0.1**2.0)])
    res = minimize(fun, x0=x0, args=(y,), options={"maxiter": 1_000}, method="L-BFGS-B")

    assert res.success, "Optimization did not converge!"

    phi_inferred = sigmoid(res.x[0])

    assert np.allclose(phi_inferred, f.transition_matrices, rtol=1e-1)
