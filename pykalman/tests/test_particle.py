"""Tests for ParticleFilter (particle.py).

Reference: Gordon, Salmond & Smith (1993).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pykalman.particle import (
    ParticleFilter,
    _systematic_resample,
    _particle_filter,
    _particle_smoother,
)


# ---------------------------------------------------------------------------
# DGP helper
# ---------------------------------------------------------------------------


def _make_ar1(T=80, phi=0.9, q=0.1, r=1.0, seed=0):
    """Simulate AR(1) state-space: x_t = phi*x_{t-1} + N(0,q), z_t = x_t + N(0,r)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((T, 1))
    for t in range(1, T):
        x[t] = phi * x[t - 1] + rng.standard_normal() * np.sqrt(q)
    z = x + rng.standard_normal((T, 1)) * np.sqrt(r)
    return x, z


def _default_pf(n_particles=500, seed=0):
    f = lambda x: 0.9 * x
    g = lambda x: x
    Q = np.array([[0.1]])
    R = np.array([[1.0]])
    mu0 = np.zeros(1)
    S0 = np.eye(1)
    return ParticleFilter(
        f, g, Q, R, mu0, S0, n_particles=n_particles, random_state=seed
    )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


def test_systematic_resample_returns_n_indices():
    rng = np.random.default_rng(0)
    weights = np.array([0.1, 0.4, 0.3, 0.2])
    idx = _systematic_resample(weights, rng)
    assert len(idx) == len(weights)


def test_systematic_resample_indices_in_range():
    rng = np.random.default_rng(1)
    N = 100
    weights = np.ones(N) / N
    idx = _systematic_resample(weights, rng)
    assert np.all(idx >= 0)
    assert np.all(idx < N)


def test_systematic_resample_deterministic_uniform():
    """Uniform weights should produce evenly spaced indices."""
    rng = np.random.default_rng(2)
    N = 10
    idx = _systematic_resample(np.ones(N) / N, rng)
    assert len(set(idx)) == N  # all unique for uniform weights


def test_systematic_resample_concentrates_on_high_weight():
    """High-weight particle should dominate resampled set."""
    rng = np.random.default_rng(3)
    N = 100
    weights = np.full(N, 1e-6)
    weights[0] = 1.0
    weights /= weights.sum()
    idx = _systematic_resample(weights, rng)
    assert np.mean(idx == 0) > 0.9


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_init_stores_params():
    pf = _default_pf()
    assert pf.n_particles == 500
    assert pf.random_state == 0
    assert pf.resampling_threshold == 0.5


def test_init_callable_transition():
    f = lambda x: x
    g = lambda x: x
    pf = ParticleFilter(f, g, np.eye(1), np.eye(1), np.zeros(1), np.eye(1))
    assert callable(pf.transition_functions)


# ---------------------------------------------------------------------------
# filter() output shapes and properties
# ---------------------------------------------------------------------------


def test_filter_output_shapes():
    pf = _default_pf()
    _, z = _make_ar1(T=40)
    means, covs = pf.filter(z)
    assert means.shape == (40, 1)
    assert covs.shape == (40, 1, 1)


def test_filter_covs_positive_semidefinite():
    pf = _default_pf(n_particles=200)
    _, z = _make_ar1(T=30)
    _, covs = pf.filter(z)
    for t in range(covs.shape[0]):
        eigvals = np.linalg.eigvalsh(covs[t])
        assert np.all(eigvals >= -1e-10), f"Cov not PSD at t={t}: {eigvals}"


def test_filter_means_finite():
    pf = _default_pf()
    _, z = _make_ar1(T=50)
    means, _ = pf.filter(z)
    assert np.all(np.isfinite(means))


def test_filter_accepts_1d_observations():
    f = lambda x: 0.9 * x
    g = lambda x: x
    pf = ParticleFilter(
        f, g, np.array([[0.1]]), np.array([[1.0]]),
        np.zeros(1), np.eye(1), n_particles=200, random_state=0,
    )
    z_1d = np.random.randn(30)  # 1-D observations
    means, covs = pf.filter(z_1d)
    assert means.shape == (30, 1)


def test_filter_mse_below_naive_for_ar1():
    """Particle filter should beat the naive zero-mean forecast."""
    x_true, z = _make_ar1(T=200, seed=7)
    pf = _default_pf(n_particles=1000, seed=7)
    means, _ = pf.filter(z)
    mse_pf = np.mean((means[:, 0] - x_true[:, 0]) ** 2)
    mse_naive = np.mean(x_true[:, 0] ** 2)
    assert mse_pf < mse_naive


# ---------------------------------------------------------------------------
# filter() accuracy on linear model vs Kalman filter
# ---------------------------------------------------------------------------


def test_filter_close_to_kalman_on_linear_model():
    """On a linear Gaussian model, PF posterior mean ≈ KF posterior mean."""
    from pykalman import KalmanFilter

    phi, q, r = 0.9, 0.1, 1.0
    x_true, z = _make_ar1(T=100, phi=phi, q=q, r=r, seed=5)

    # Particle filter
    f = lambda x: phi * x
    g = lambda x: x
    Q = np.array([[q]])
    R = np.array([[r]])
    mu0, S0 = np.zeros(1), np.array([[1.0]])
    pf = ParticleFilter(f, g, Q, R, mu0, S0, n_particles=3000, random_state=5)
    pf_means, _ = pf.filter(z)

    # Standard Kalman filter
    kf = KalmanFilter(
        transition_matrices=np.array([[phi]]),
        observation_matrices=np.eye(1),
        transition_covariance=Q,
        observation_covariance=R,
        initial_state_mean=mu0,
        initial_state_covariance=S0,
    )
    kf_means, _ = kf.filter(z)

    # Monte Carlo noise so allow generous tolerance
    assert_allclose(pf_means[:, 0], kf_means[:, 0], atol=0.35)


# ---------------------------------------------------------------------------
# smooth() output shapes and properties
# ---------------------------------------------------------------------------


def test_smooth_output_shapes():
    pf = _default_pf(n_particles=300)
    _, z = _make_ar1(T=40)
    s_means, s_covs = pf.smooth(z)
    assert s_means.shape == (40, 1)
    assert s_covs.shape == (40, 1, 1)


def test_smooth_means_finite():
    pf = _default_pf(n_particles=200)
    _, z = _make_ar1(T=30)
    s_means, _ = pf.smooth(z)
    assert np.all(np.isfinite(s_means))


def test_smooth_mse_leq_filter_mse():
    """Smoother should not be worse than the filter (on average)."""
    x_true, z = _make_ar1(T=150, seed=11)
    pf = ParticleFilter(
        lambda x: 0.9 * x, lambda x: x,
        np.array([[0.1]]), np.array([[1.0]]),
        np.zeros(1), np.eye(1),
        n_particles=1000, random_state=11,
    )
    f_means, _ = pf.filter(z)
    s_means, _ = pf.smooth(z)
    mse_f = np.mean((f_means[:, 0] - x_true[:, 0]) ** 2)
    mse_s = np.mean((s_means[:, 0] - x_true[:, 0]) ** 2)
    assert mse_s <= mse_f + 0.05  # small slack for MC noise


# ---------------------------------------------------------------------------
# sample()
# ---------------------------------------------------------------------------


def test_sample_output_shapes():
    pf = _default_pf()
    states, obs = pf.sample(50)
    assert states.shape == (50, 1)
    assert obs.shape == (50, 1)


def test_sample_states_finite():
    pf = _default_pf()
    states, obs = pf.sample(30)
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(obs))


def test_sample_with_initial_state():
    pf = _default_pf()
    states, _ = pf.sample(20, initial_state=np.array([5.0]))
    assert states[0, 0] == pytest.approx(5.0)


def test_sample_reproducible():
    pf = _default_pf(seed=99)
    s1, o1 = pf.sample(30, random_state=42)
    s2, o2 = pf.sample(30, random_state=42)
    assert_allclose(s1, s2)
    assert_allclose(o1, o2)


# ---------------------------------------------------------------------------
# Time-varying functions
# ---------------------------------------------------------------------------


def test_filter_list_of_functions():
    """Per-timestep transition functions should be accepted."""
    T = 20
    f_list = [lambda x: 0.9 * x] * T
    g_list = [lambda x: x] * T
    pf = ParticleFilter(
        f_list, g_list, np.eye(1) * 0.1, np.eye(1),
        np.zeros(1), np.eye(1), n_particles=200, random_state=0,
    )
    _, z = _make_ar1(T=T)
    means, _ = pf.filter(z)
    assert means.shape == (T, 1)


# ---------------------------------------------------------------------------
# Multivariate state
# ---------------------------------------------------------------------------


def test_filter_2d_state():
    """2-D state, 1-D observation."""
    T, n_state, n_obs = 50, 2, 1
    A = np.array([[0.9, 0.1], [0.0, 0.8]])
    H = np.array([[1.0, 0.0]])
    f = lambda x: A @ x
    g = lambda x: H @ x
    Q = np.eye(2) * 0.05
    R = np.eye(1) * 0.5
    mu0 = np.zeros(2)
    S0 = np.eye(2)

    rng = np.random.default_rng(3)
    x = np.zeros((T, 2))
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.multivariate_normal(np.zeros(2), Q)
    z = (x @ H.T) + rng.standard_normal((T, 1)) * np.sqrt(0.5)

    pf = ParticleFilter(
        f, g, Q, R, mu0, S0, n_particles=500, random_state=3
    )
    means, covs = pf.filter(z)
    assert means.shape == (T, 2)
    assert covs.shape == (T, 2, 2)
    assert np.all(np.isfinite(means))


# ---------------------------------------------------------------------------
# Nonlinear model — particle filter advantage
# ---------------------------------------------------------------------------


def test_filter_nonlinear_model():
    """Nonlinear model: x_t = cos(x_{t-1}), z_t = x_t + noise."""
    T = 40
    f = lambda x: np.cos(x)
    g = lambda x: x
    Q = np.array([[0.1]])
    R = np.array([[0.5]])
    mu0 = np.zeros(1)
    S0 = np.eye(1)

    rng = np.random.default_rng(8)
    x = np.zeros((T, 1))
    for t in range(1, T):
        x[t] = np.cos(x[t - 1]) + rng.standard_normal() * np.sqrt(0.1)
    z = x + rng.standard_normal((T, 1)) * np.sqrt(0.5)

    pf = ParticleFilter(f, g, Q, R, mu0, S0, n_particles=500, random_state=8)
    means, covs = pf.filter(z)
    assert means.shape == (T, 1)
    assert np.all(np.isfinite(means))
    mse = np.mean((means[:, 0] - x[:, 0]) ** 2)
    assert mse < 0.5


# ---------------------------------------------------------------------------
# get_params
# ---------------------------------------------------------------------------


def test_get_params_contains_key_attributes():
    pf = _default_pf()
    p = pf.get_params()
    assert "n_particles" in p
    assert "resampling_threshold" in p
