"""Particle Filter (Sequential Importance Resampling).

============================================
Inference for Nonlinear Non-Gaussian Systems
============================================

This module implements the bootstrap Particle Filter (Gordon, Salmond &
Smith 1993) for state estimation in systems with nonlinear dynamics and
non-Gaussian noise, together with a particle backward-simulation smoother
(Godsill, Doucet & West 2004).

The state-space model assumes *additive* Gaussian noise:

    x_0   ~  Normal(mu_0, Sigma_0)
    x_t   =  f_t(x_{t-1})  +  Normal(0, Q)     (transition)
    z_t   =  g_t(x_t)      +  Normal(0, R)     (observation)

For fully general (non-additive) noise use the UnscentedKalmanFilter or
extend this module.

References
----------
Gordon, N. J., Salmond, D. J., & Smith, A. F. M. (1993). Novel approach to
nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F*, 140(2),
107-113.

Godsill, S. J., Doucet, A., & West, M. (2004). Monte Carlo smoothing for
nonlinear time series. *Journal of the American Statistical Association*,
99(465), 156-168.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from .utils import array1d, array2d, check_random_state, get_params
from .standard import _arg_or_default, _determine_dimensionality


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _systematic_resample(weights, rng):
    """Systematic resampling — lower variance than multinomial sampling.

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Normalised importance weights.
    rng : RandomState

    Returns
    -------
    indices : ndarray of int, shape (N,)
    """
    N = len(weights)
    positions = (rng.uniform() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    return np.searchsorted(cumsum, positions)


def _log_obs_weight(obs, pred_obs, R_inv, log_det_R):
    """Log N(obs; pred_obs, R) for a single particle."""
    n = len(obs)
    diff = obs - pred_obs
    return -0.5 * (diff @ R_inv @ diff + log_det_R + n * np.log(2.0 * np.pi))


def _particle_filter(
    Z,
    transition_functions,
    observation_functions,
    Q,
    R,
    mu_0,
    Sigma_0,
    n_particles,
    resampling_threshold,
    rng,
    store_particles=False,
):
    """Core SIR particle filter.

    Parameters
    ----------
    Z : ndarray, shape (T, n_dim_obs)
    transition_functions : list of T callables  f_t: (n_dim_state,) -> (n_dim_state,)
    observation_functions : list of T callables  g_t: (n_dim_state,) -> (n_dim_obs,)
    Q : ndarray, shape (n_dim_state, n_dim_state)  transition covariance
    R : ndarray, shape (n_dim_obs, n_dim_obs)  observation covariance
    mu_0 : ndarray, shape (n_dim_state,)
    Sigma_0 : ndarray, shape (n_dim_state, n_dim_state)
    n_particles : int
    resampling_threshold : float
    rng : RandomState
    store_particles : bool
        If True, return all particle clouds (needed for smoothing).

    Returns
    -------
    means : ndarray, shape (T, n_dim_state)
    covs : ndarray, shape (T, n_dim_state, n_dim_state)
    particles_all : list of length T of ndarray shape (N, n_dim_state), or None
    weights_all : list of length T of ndarray shape (N,), or None
    """
    T = Z.shape[0]
    N = n_particles
    n_state = len(mu_0)

    # Precompute observation model constants.
    R_inv = np.linalg.inv(R)
    log_det_R = np.log(np.linalg.det(R))

    # Cholesky for sampling from Q.
    Q_chol = np.linalg.cholesky(Q)

    # Initialise particles.
    particles = rng.multivariate_normal(mu_0, Sigma_0, size=N)  # (N, n_state)
    log_weights = np.full(N, -np.log(N))                        # uniform

    means = np.empty((T, n_state))
    covs = np.empty((T, n_state, n_state))
    particles_all = [] if store_particles else None
    weights_all = [] if store_particles else None

    for t in range(T):
        f_t = transition_functions[t]
        g_t = observation_functions[t]

        # --- Propagate through transition ---
        propagated = np.empty_like(particles)
        for i in range(N):
            propagated[i] = f_t(particles[i])
        noise = rng.standard_normal((N, n_state)) @ Q_chol.T
        particles = propagated + noise

        # --- Observation update (log-weight increment) ---
        pred_obs = np.empty((N, Z.shape[1]))
        for i in range(N):
            pred_obs[i] = g_t(particles[i])

        diff = Z[t] - pred_obs                           # (N, n_obs)
        log_liks = -0.5 * (
            np.einsum("ni,ij,nj->n", diff, R_inv, diff)
            + log_det_R
            + Z.shape[1] * np.log(2.0 * np.pi)
        )
        log_weights = log_weights + log_liks

        # Numerically stable normalisation.
        log_weights -= special.logsumexp(log_weights)
        weights = np.exp(log_weights)

        # --- State estimate ---
        means[t] = weights @ particles
        diff_state = particles - means[t]                # (N, n_state)
        covs[t] = (weights[:, None] * diff_state).T @ diff_state

        # --- Adaptive resampling ---
        ess = 1.0 / np.sum(weights ** 2)
        if ess < resampling_threshold * N:
            idx = _systematic_resample(weights, rng)
            particles = particles[idx]
            log_weights = np.full(N, -np.log(N))
            weights = np.full(N, 1.0 / N)

        if store_particles:
            particles_all.append(particles.copy())
            weights_all.append(weights.copy())

    return means, covs, particles_all, weights_all


def _particle_smoother(
    particles_all,
    weights_all,
    transition_functions,
    Q,
    n_smooth,
    rng,
):
    """Particle backward-simulation smoother (Godsill, Doucet & West 2004).

    Draws ``n_smooth`` complete trajectories from the joint smoothing
    distribution using backward kernel sampling, then summarises them with
    weighted sample mean and covariance.

    Parameters
    ----------
    particles_all : list of T arrays, each (N, n_dim_state)
    weights_all : list of T arrays, each (N,)
    transition_functions : list of T callables
    Q : ndarray, shape (n_dim_state, n_dim_state)
    n_smooth : int
        Number of backward trajectories to sample.
    rng : RandomState

    Returns
    -------
    means : ndarray, shape (T, n_dim_state)
    covs : ndarray, shape (T, n_dim_state, n_dim_state)
    """
    T = len(particles_all)
    N = particles_all[0].shape[0]
    n_state = particles_all[0].shape[1]

    Q_inv = np.linalg.inv(Q)
    log_det_Q = np.log(np.linalg.det(Q))
    n_smooth = min(n_smooth, N)

    # Trajectories: shape (T, n_smooth, n_state).
    trajectories = np.empty((T, n_smooth, n_state))

    # --- Sample terminal particles proportional to filtering weights ---
    w_T = weights_all[-1]
    term_idx = rng.choice(N, size=n_smooth, p=w_T / w_T.sum(), replace=True)
    trajectories[-1] = particles_all[-1][term_idx]

    # --- Backward pass ---
    for t in range(T - 2, -1, -1):
        f_next = transition_functions[t + 1]
        filt_particles = particles_all[t]     # (N, n_state)
        log_w_t = np.log(weights_all[t] + 1e-300)

        # Propagate all filter particles forward once for the kernel.
        propagated = np.array([f_next(p) for p in filt_particles])  # (N, n_state)

        for b in range(n_smooth):
            x_next = trajectories[t + 1, b]           # (n_state,)
            diff = x_next - propagated                  # (N, n_state)
            log_bk = -0.5 * (
                np.einsum("ni,ij,nj->n", diff, Q_inv, diff)
                + log_det_Q
                + n_state * np.log(2.0 * np.pi)
            )
            log_bk += log_w_t
            log_bk -= special.logsumexp(log_bk)
            bk = np.exp(log_bk)
            anc = rng.choice(N, p=bk)
            trajectories[t, b] = filt_particles[anc]

    means = trajectories.mean(axis=1)
    covs = np.array([np.atleast_2d(np.cov(trajectories[t].T))
                     if n_smooth > 1 else np.zeros((n_state, n_state))
                     for t in range(T)])
    return means, covs


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class ParticleFilter:
    r"""Bootstrap Particle Filter with backward-simulation smoother.

    Estimates the latent state sequence for a nonlinear, potentially
    non-Gaussian discrete-time state-space model with *additive* Gaussian
    noise:

    .. math::

        x_0   &\sim \mathcal{N}(\mu_0, \Sigma_0) \\
        x_t   &= f_t(x_{t-1}) + \mathcal{N}(0, Q) \\
        z_t   &= g_t(x_t)     + \mathcal{N}(0, R)

    The filter uses Sequential Importance Resampling (SIR): at each step
    :math:`N` particles are propagated through :math:`f_t`, importance-
    weighted by the observation likelihood, and resampled when the
    effective sample size falls below ``resampling_threshold * n_particles``.

    Smoothing is performed by drawing backward trajectories from the joint
    smoothing distribution using the backward simulation kernel of Godsill,
    Doucet & West (2004).

    Parameters
    ----------
    transition_functions : callable or list of T callables
        ``f_t(x)`` maps a state vector to the next expected state.
        A single callable is broadcast over all time steps.
    observation_functions : callable or list of T callables
        ``g_t(x)`` maps a state vector to the expected observation.
    transition_covariance : array_like, shape (n_dim_state, n_dim_state)
        Covariance matrix :math:`Q` of the additive state transition noise.
    observation_covariance : array_like, shape (n_dim_obs, n_dim_obs)
        Covariance matrix :math:`R` of the additive observation noise.
    initial_state_mean : array_like, shape (n_dim_state,)
        Mean :math:`\mu_0` of the initial state distribution.
    initial_state_covariance : array_like, shape (n_dim_state, n_dim_state)
        Covariance :math:`\Sigma_0` of the initial state distribution.
    n_particles : int, default 1000
        Number of particles :math:`N`.
    resampling_threshold : float in (0, 1), default 0.5
        Resample when :math:`\text{ESS} < \text{threshold} \times N`.
    n_smooth : int or None, default None
        Number of backward trajectories used in :meth:`smooth`.  Defaults
        to ``n_particles``.
    random_state : int, RandomState, or None
        Seed for reproducible sampling.

    References
    ----------
    Gordon, N. J., Salmond, D. J., & Smith, A. F. M. (1993). Novel approach
    to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F
    (Radar and Signal Processing)*, 140(2), 107-113.

    Godsill, S. J., Doucet, A., & West, M. (2004). Monte Carlo smoothing for
    nonlinear time series. *Journal of the American Statistical Association*,
    99(465), 156-168.

    Examples
    --------
    >>> import numpy as np
    >>> from pykalman.particle import ParticleFilter
    >>> f = lambda x: 0.9 * x          # linear transition
    >>> g = lambda x: x                # identity observation
    >>> Q = np.eye(1) * 0.1
    >>> R = np.eye(1) * 1.0
    >>> mu0, S0 = np.zeros(1), np.eye(1)
    >>> pf = ParticleFilter(f, g, Q, R, mu0, S0, n_particles=500,
    ...                     random_state=0)
    >>> Z = np.random.randn(50, 1)
    >>> means, covs = pf.filter(Z)
    >>> means.shape
    (50, 1)
    """

    def __init__(
        self,
        transition_functions,
        observation_functions,
        transition_covariance,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        n_particles=1000,
        resampling_threshold=0.5,
        n_smooth=None,
        random_state=None,
    ):
        self.transition_functions = transition_functions
        self.observation_functions = observation_functions
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.n_particles = n_particles
        self.resampling_threshold = resampling_threshold
        self.n_smooth = n_smooth
        self.random_state = random_state

    def _parse_params(self, Z):
        """Validate and broadcast parameters to per-timestep lists."""
        T = Z.shape[0]
        Q = array2d(self.transition_covariance)
        R = array2d(self.observation_covariance)
        mu_0 = array1d(self.initial_state_mean)
        S_0 = array2d(self.initial_state_covariance)

        def _broadcast(fns, T):
            if callable(fns):
                return [fns] * T
            fns = list(fns)
            if len(fns) == 1:
                return fns * T
            if len(fns) != T:
                raise ValueError(
                    f"Expected {T} functions, got {len(fns)}."
                )
            return fns

        f_list = _broadcast(self.transition_functions, T)
        g_list = _broadcast(self.observation_functions, T)
        return f_list, g_list, Q, R, mu_0, S_0

    def filter(self, Z):
        """Apply the SIR particle filter to an observation sequence.

        Parameters
        ----------
        Z : array_like, shape (T, n_dim_obs)
            Observation sequence.

        Returns
        -------
        filtered_state_means : ndarray, shape (T, n_dim_state)
            Particle-weighted posterior mean at each time step.
        filtered_state_covs : ndarray, shape (T, n_dim_state, n_dim_state)
            Particle-weighted posterior covariance at each time step.
        """
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z[:, None]

        rng = check_random_state(self.random_state)
        f_list, g_list, Q, R, mu_0, S_0 = self._parse_params(Z)

        means, covs, _, _ = _particle_filter(
            Z, f_list, g_list, Q, R, mu_0, S_0,
            self.n_particles, self.resampling_threshold, rng,
            store_particles=False,
        )
        return means, covs

    def smooth(self, Z):
        """Apply the particle backward-simulation smoother.

        Runs the forward particle filter, then samples backward trajectories
        to estimate the joint smoothing distribution.

        Parameters
        ----------
        Z : array_like, shape (T, n_dim_obs)
            Observation sequence.

        Returns
        -------
        smoothed_state_means : ndarray, shape (T, n_dim_state)
            Monte Carlo estimate of the posterior mean under the full
            observation sequence.
        smoothed_state_covs : ndarray, shape (T, n_dim_state, n_dim_state)
            Monte Carlo estimate of the posterior covariance.
        """
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z[:, None]

        rng = check_random_state(self.random_state)
        f_list, g_list, Q, R, mu_0, S_0 = self._parse_params(Z)

        _, _, particles_all, weights_all = _particle_filter(
            Z, f_list, g_list, Q, R, mu_0, S_0,
            self.n_particles, self.resampling_threshold, rng,
            store_particles=True,
        )

        n_smooth = self.n_smooth if self.n_smooth is not None else self.n_particles
        s_means, s_covs = _particle_smoother(
            particles_all, weights_all, f_list, Q, n_smooth, rng,
        )
        return s_means, s_covs

    def sample(self, n_timesteps, initial_state=None, random_state=None):
        """Sample a state and observation sequence from the model.

        Parameters
        ----------
        n_timesteps : int
        initial_state : array_like, shape (n_dim_state,) or None
            Starting state.  Sampled from the prior if None.
        random_state : int or RandomState or None

        Returns
        -------
        states : ndarray, shape (n_timesteps, n_dim_state)
        observations : ndarray, shape (n_timesteps, n_dim_obs)
        """
        rng = check_random_state(
            random_state if random_state is not None else self.random_state
        )
        Q = array2d(self.transition_covariance)
        R = array2d(self.observation_covariance)
        mu_0 = array1d(self.initial_state_mean)
        S_0 = array2d(self.initial_state_covariance)

        f_list, g_list, _, _, _, _ = self._parse_params(
            np.zeros((n_timesteps, R.shape[0]))
        )

        n_state = len(mu_0)
        n_obs = R.shape[0]
        Q_chol = np.linalg.cholesky(Q)
        R_chol = np.linalg.cholesky(R)

        if initial_state is None:
            x = rng.multivariate_normal(mu_0, S_0)
        else:
            x = np.asarray(initial_state, dtype=float)

        states = np.empty((n_timesteps, n_state))
        observations = np.empty((n_timesteps, n_obs))

        for t in range(n_timesteps):
            states[t] = x
            observations[t] = g_list[t](x) + R_chol @ rng.standard_normal(n_obs)
            x = f_list[t](x) + Q_chol @ rng.standard_normal(n_state)

        return states, observations

    def get_params(self):
        """Return model parameters as a dict."""
        return get_params(self)
