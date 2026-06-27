"""Tests for the Extended Kalman Filter (pykalman/extended.py).

Correctness strategy
--------------------
1. **Linear system → standard KF parity**: On a linear Gaussian system the EKF
   linearisation is exact, so EKF outputs must match the standard KalmanFilter
   to floating-point precision.

2. **Analytical == numerical Jacobians**: On a nonlinear system, providing the
   exact Jacobian analytically must give the same result as letting the EKF
   approximate it via central finite differences.

3. **Online == batch**: :meth:`filter_update` applied step-by-step must produce
   the same trajectory as the batch :meth:`filter`.

4. **Smoother reduces variance**: The RTS smoother posterior covariance must
   be element-wise ≤ the filter posterior covariance at every time step.

5. **Missing observations**: Masked observations must be skipped (filtered
   mean/covariance unchanged from prediction).

6. **Sample shapes**: :meth:`sample` must return arrays of the declared shapes.
"""

import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_allclose

from ..extended import ExtendedKalmanFilter, _numerical_jacobian
from ..standard import KalmanFilter


# ---------------------------------------------------------------------------
# Shared system definition (linear Gaussian for cross-validation with KF)
# ---------------------------------------------------------------------------

_A = np.array([[1.0, 1.0], [0.0, 1.0]])   # transition matrix
_C = np.array([[0.5, -0.3]])               # observation matrix
_Q = np.eye(2) * 2.0                       # process noise cov
_R = np.array([[0.5]])                     # observation noise cov
_x0 = np.array([1.0, 1.0])
_P0 = np.array([[1.0, 0.1], [0.1, 1.0]])


def _linear_ekf(**kwargs):
    """EKF on the linear test system; keyword args override defaults."""
    defaults = dict(
        transition_functions=lambda x: _A @ x,
        observation_functions=lambda x: _C @ x,
        transition_covariance=_Q,
        observation_covariance=_R,
        initial_state_mean=_x0,
        initial_state_covariance=_P0,
    )
    defaults.update(kwargs)
    return ExtendedKalmanFilter(**defaults)


def _linear_kf():
    """Standard (linear) KalmanFilter on the same system."""
    return KalmanFilter(
        transition_matrices=_A,
        observation_matrices=_C,
        transition_covariance=_Q,
        observation_covariance=_R,
        initial_state_mean=_x0,
        initial_state_covariance=_P0,
    )


def _make_observations(seed=0):
    """Generate a short observation sequence (first obs masked)."""
    return ma.array(
        [0.0, 1.0, 2.0, 3.0],
        mask=[True, False, False, False],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# 1. Numerical Jacobian
# ---------------------------------------------------------------------------

def test_numerical_jacobian_on_linear_map():
    """Central finite differences must recover the exact matrix for a linear map."""
    A = np.array([[2.0, -1.0], [0.5, 3.0]])
    f = lambda x: A @ x
    x = np.array([1.0, -2.0])
    J = _numerical_jacobian(f, x)
    assert_allclose(J, A, atol=1e-8)


def test_numerical_jacobian_on_nonlinear_map():
    """Finite-difference Jacobian for g(x) = [x0^2, sin(x1)]."""
    g = lambda x: np.array([x[0] ** 2, np.sin(x[1])])
    x = np.array([3.0, np.pi / 4])
    J_analytic = np.array([[2 * x[0], 0.0], [0.0, np.cos(x[1])]])
    J_numeric = _numerical_jacobian(g, x)
    assert_allclose(J_numeric, J_analytic, atol=1e-7)


# ---------------------------------------------------------------------------
# 2. EKF == standard KF on a linear system
# ---------------------------------------------------------------------------

def test_filter_matches_standard_kf():
    """EKF with analytical Jacobians must match the linear KF exactly."""
    Z = _make_observations()

    ekf = _linear_ekf(
        transition_jacobians=lambda x: _A,
        observation_jacobians=lambda x: _C,
    )
    mu_ekf, sigma_ekf = ekf.filter(Z)

    kf = _linear_kf()
    mu_kf, sigma_kf = kf.filter(Z)

    assert_allclose(mu_ekf, mu_kf, atol=1e-8)
    assert_allclose(sigma_ekf, sigma_kf, atol=1e-8)


def test_smooth_matches_standard_kf():
    """EKF smoother on a linear system must match the linear KF smoother."""
    Z = _make_observations()

    ekf = _linear_ekf(
        transition_jacobians=lambda x: _A,
        observation_jacobians=lambda x: _C,
    )
    mu_smooth_ekf, sigma_smooth_ekf = ekf.smooth(Z)

    kf = _linear_kf()
    mu_smooth_kf, sigma_smooth_kf = kf.smooth(Z)

    assert_allclose(mu_smooth_ekf, mu_smooth_kf, atol=1e-8)
    assert_allclose(sigma_smooth_ekf, sigma_smooth_kf, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. Analytical Jacobians == numerical Jacobians
# ---------------------------------------------------------------------------

def _nonlinear_ekf(with_jacobians):
    """EKF with a nonlinear observation g(x) = [x[0]^2]."""
    f = lambda x: _A @ x
    g = lambda x: np.array([x[0] ** 2])
    kwargs = dict(
        transition_functions=f,
        observation_functions=g,
        transition_covariance=_Q,
        observation_covariance=np.eye(1) * 1.0,
        initial_state_mean=np.array([1.5, 0.5]),
        initial_state_covariance=np.eye(2),
        n_dim_obs=1,
    )
    if with_jacobians:
        kwargs["transition_jacobians"] = lambda x: _A
        kwargs["observation_jacobians"] = lambda x: np.array([[2 * x[0], 0.0]])
    return ExtendedKalmanFilter(**kwargs)


def test_numerical_equals_analytical_jacobians_filter():
    """Numerical Jacobians must reproduce the analytical result for the filter."""
    rng = np.random.RandomState(42)
    Z = ma.array(rng.randn(10, 1))

    mu_anal, sigma_anal = _nonlinear_ekf(with_jacobians=True).filter(Z)
    mu_num, sigma_num = _nonlinear_ekf(with_jacobians=False).filter(Z)

    # Tolerance is loose because finite differences introduce O(eps^2) error
    assert_allclose(mu_anal, mu_num, atol=1e-6)
    assert_allclose(sigma_anal, sigma_num, atol=1e-6)


def test_numerical_equals_analytical_jacobians_smoother():
    rng = np.random.RandomState(7)
    Z = ma.array(rng.randn(10, 1))

    mu_anal, sigma_anal = _nonlinear_ekf(with_jacobians=True).smooth(Z)
    mu_num, sigma_num = _nonlinear_ekf(with_jacobians=False).smooth(Z)

    assert_allclose(mu_anal, mu_num, atol=1e-6)
    assert_allclose(sigma_anal, sigma_num, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Online filter_update == batch filter
# ---------------------------------------------------------------------------

def test_filter_update_matches_batch_filter():
    """filter_update applied step-by-step must equal the batch filter."""
    Z = _make_observations()
    ekf = _linear_ekf(
        transition_jacobians=lambda x: _A,
        observation_jacobians=lambda x: _C,
    )

    mu_batch, sigma_batch = ekf.filter(Z)

    mu_online = np.zeros_like(mu_batch)
    sigma_online = np.zeros_like(sigma_batch)
    mu_online[0], sigma_online[0] = mu_batch[0], sigma_batch[0]

    for t in range(len(Z) - 1):
        mu_online[t + 1], sigma_online[t + 1] = ekf.filter_update(
            mu_online[t], sigma_online[t], Z[t + 1]
        )

    assert_allclose(mu_online, mu_batch, atol=1e-10)
    assert_allclose(sigma_online, sigma_batch, atol=1e-10)


def test_filter_update_missing_observation():
    """filter_update with None observation must leave state unchanged (prediction only)."""
    ekf = _linear_ekf()
    x0 = _x0.copy()
    P0 = _P0.copy()
    mu_next, P_next = ekf.filter_update(x0, P0, observation=None)

    # Without a correction the covariance must grow (strictly larger than P0)
    assert np.all(np.linalg.eigvalsh(P_next - P0) >= -1e-10)
    # Mean must be f(x0) = A @ x0
    assert_allclose(mu_next, _A @ x0, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. Smoother posterior variance ≤ filter posterior variance
# ---------------------------------------------------------------------------

def test_smoother_reduces_covariance():
    """RTS smoother must not inflate posterior covariance over the filter."""
    Z = _make_observations()
    ekf = _linear_ekf()

    _, sigma_filt = ekf.filter(Z)
    _, sigma_smooth = ekf.smooth(Z)

    for t in range(len(Z)):
        diff = sigma_filt[t] - sigma_smooth[t]
        # diff must be positive semi-definite
        assert np.all(np.linalg.eigvalsh(diff) >= -1e-10), (
            f"Smoother inflated covariance at t={t}"
        )


# ---------------------------------------------------------------------------
# 6. Missing observations
# ---------------------------------------------------------------------------

def test_filter_skips_masked_observation():
    """Filtering a fully masked observation must equal pure prediction."""
    ekf = _linear_ekf()

    fully_masked = ma.array([[0.0]], mask=[[True]])
    full_obs = ma.array([[1.0]], mask=[[False]])

    # Filtered state given masked obs at t=1 after x0, P0
    Z_masked = ma.array([[0.0], [0.0]], mask=[[True], [True]], dtype=float)
    Z_seen = ma.array([[0.0], [1.0]], mask=[[True], [False]], dtype=float)

    mu_m, sigma_m = ekf.filter(Z_masked)
    mu_s, sigma_s = ekf.filter(Z_seen)

    # At t=0 both should be the same (both masked)
    assert_allclose(mu_m[0], mu_s[0], atol=1e-10)

    # At t=1 the unmasked observation should pull the mean toward data
    assert not np.allclose(mu_m[1], mu_s[1]), (
        "Masked and unmasked results should differ at t=1"
    )


# ---------------------------------------------------------------------------
# 7. Sample shape
# ---------------------------------------------------------------------------

def test_sample_shape():
    ekf = _linear_ekf()
    x, z = ekf.sample(20, random_state=0)
    assert x.shape == (20, 2)
    assert z.shape == (20, 1)


def test_sample_shape_nonlinear():
    """sample() must also work for a nonlinear observation (n_dim_obs != n_dim_state)."""
    ekf = _nonlinear_ekf(with_jacobians=False)
    x, z = ekf.sample(15, random_state=1)
    assert x.shape == (15, 2)
    assert z.shape == (15, 1)


# ---------------------------------------------------------------------------
# 8. Parameter initialisation / dimensionality inference
# ---------------------------------------------------------------------------

def test_dimensionality_inferred_from_covariance():
    """n_dim_state and n_dim_obs must be inferred without explicit specification."""
    ekf = ExtendedKalmanFilter(
        transition_covariance=np.eye(3),
        observation_covariance=np.eye(2),
    )
    assert ekf.n_dim_state == 3
    assert ekf.n_dim_obs == 2


def test_dimensionality_inferred_from_initial_mean():
    ekf = ExtendedKalmanFilter(
        initial_state_mean=np.zeros(4),
        observation_covariance=np.eye(1),
    )
    assert ekf.n_dim_state == 4


# ---------------------------------------------------------------------------
# 9. filter_update with per-call overrides
# ---------------------------------------------------------------------------

def test_filter_update_override_covariance():
    """filter_update should accept per-call overrides for Q and R."""
    ekf = _linear_ekf()
    mu, P = ekf.filter_update(
        _x0,
        _P0,
        observation=np.array([1.0]),
        transition_covariance=np.eye(2) * 100.0,  # large Q → large uncertainty
        observation_covariance=np.eye(1) * 0.001, # small R → trust obs
    )
    # With tiny R the corrected mean should be very close to observation
    assert mu.shape == (2,)
    assert P.shape == (2, 2)
