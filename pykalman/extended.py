"""Extended Kalman Filter.

==========================================
Inference for Non-Linear Gaussian Systems
==========================================

This module implements the Extended Kalman Filter (EKF) and its companion
Rauch-Tung-Striebel smoother for state-space models with **additive** Gaussian
noise and nonlinear dynamics:

.. math::

    x_0       &\\sim \\mathcal{N}(\\mu_0, \\Sigma_0)        \\\\
    x_{t+1}   &=    f_t(x_t) + \\mathcal{N}(0, Q)        \\\\
    z_t       &=    g_t(x_t) + \\mathcal{N}(0, R)

The EKF linearises :math:`f_t` and :math:`g_t` via first-order Taylor
expansion (Jacobians) around the current state estimate.  Jacobians may be
supplied analytically or omitted, in which case they are approximated
automatically via central finite differences.

References
----------
Simon, D. (2006). Optimal State Estimation. John Wiley & Sons.

Sarkka, S. (2013). Bayesian Filtering and Smoothing. Cambridge University
Press.
"""

import numpy as np
from numpy import ma
from scipy import linalg

from .standard import _determine_dimensionality, _last_dims
from .utils import (
    array1d,
    array2d,
    check_random_state,
    get_params,
    preprocess_arguments,
)
from .utils_numpy import newbyteorder

_FD_EPS = 1e-5  # central finite-difference step size


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _numerical_jacobian(f, x, eps=_FD_EPS):
    """Jacobian of ``f`` at ``x`` via central finite differences.

    Parameters
    ----------
    f : callable, [n] -> [m]
        Vector-valued function.
    x : [n] array
        Point of evaluation.
    eps : float
        Step size.

    Returns
    -------
    J : [m, n] array
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    y0 = np.atleast_1d(f(x))
    m = y0.size
    J = np.empty((m, n))
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = eps
        J[:, j] = (
            np.atleast_1d(f(x + dx)) - np.atleast_1d(f(x - dx))
        ) / (2.0 * eps)
    return J


def _get_jacobian(jac_fn, f, x):
    """Return Jacobian matrix.

    If *jac_fn* is not None, return ``np.atleast_2d(jac_fn(x))``.
    Otherwise fall back to :func:`_numerical_jacobian`.
    """
    if jac_fn is not None:
        return np.atleast_2d(jac_fn(x))
    return _numerical_jacobian(f, x)


# ---------------------------------------------------------------------------
# Core filter / smoother routines
# ---------------------------------------------------------------------------

def _ekf_predict(f_t, F_t, x_filt, P_filt, Q):
    """EKF time-update (prediction) step.

    Parameters
    ----------
    f_t : callable [n] -> [n]
        State transition function.
    F_t : callable [n] -> [n, n] or None
        Jacobian of *f_t*; computed numerically when None.
    x_filt : [n_dim_state] array
        Filtered state mean at time *t*.
    P_filt : [n_dim_state, n_dim_state] array
        Filtered state covariance at time *t*.
    Q : [n_dim_state, n_dim_state] array
        Process noise covariance.

    Returns
    -------
    x_pred : [n_dim_state] array
    P_pred : [n_dim_state, n_dim_state] array
    Ft : [n_dim_state, n_dim_state] array
        Evaluated Jacobian (stored for the RTS smoother).
    """
    x_pred = np.atleast_1d(f_t(x_filt))
    Ft = _get_jacobian(F_t, f_t, x_filt)
    P_pred = Ft @ P_filt @ Ft.T + Q
    return x_pred, P_pred, Ft


def _ekf_correct(g_t, H_t, x_pred, P_pred, R, z):
    """EKF measurement-update (correction) step.

    Parameters
    ----------
    g_t : callable [n] -> [m]
        Observation function.
    H_t : callable [n] -> [m, n] or None
        Jacobian of *g_t*; computed numerically when None.
    x_pred : [n_dim_state] array
    P_pred : [n_dim_state, n_dim_state] array
    R : [n_dim_obs, n_dim_obs] array
        Observation noise covariance.
    z : [n_dim_obs] array or masked array
        Observation.  A fully masked *z* causes the correction to be skipped.

    Returns
    -------
    x_filt : [n_dim_state] array
    P_filt : [n_dim_state, n_dim_state] array
    K : [n_dim_state, n_dim_obs] array
        Kalman gain (all zeros when *z* is missing).
    Ht : [n_dim_obs, n_dim_state] array
        Evaluated observation Jacobian.
    """
    Ht = _get_jacobian(H_t, g_t, x_pred)

    if not np.any(ma.getmask(z)):
        z_pred = np.atleast_1d(g_t(x_pred))
        S = Ht @ P_pred @ Ht.T + R
        K = P_pred @ Ht.T @ linalg.pinv(S)
        x_filt = x_pred + K @ (np.asarray(z) - z_pred)
        # Joseph form: numerically stable covariance update
        n = x_pred.shape[0]
        I_KH = np.eye(n) - K @ Ht
        P_filt = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
    else:
        n_obs = R.shape[0]
        K = np.zeros((len(x_pred), n_obs))
        x_filt = x_pred.copy()
        P_filt = P_pred.copy()

    return x_filt, P_filt, K, Ht


def _extended_kalman_filter(
    transition_functions,
    observation_functions,
    transition_jacobians,
    observation_jacobians,
    transition_covariance,
    observation_covariance,
    initial_state_mean,
    initial_state_covariance,
    observations,
):
    """Apply the Extended Kalman Filter.

    Parameters
    ----------
    transition_functions : [n_timesteps-1] array of callables
    observation_functions : [n_timesteps] array of callables
    transition_jacobians : [n_timesteps-1] array of callables or None
    observation_jacobians : [n_timesteps] array of callables or None
    transition_covariance : [n_dim_state, n_dim_state] array
    observation_covariance : [n_dim_obs, n_dim_obs] array
    initial_state_mean : [n_dim_state] array
    initial_state_covariance : [n_dim_state, n_dim_state] array
    observations : [n_timesteps, n_dim_obs] masked array

    Returns
    -------
    predicted_state_means : [n_timesteps, n_dim_state] array
    predicted_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
    kalman_gains : [n_timesteps, n_dim_state, n_dim_obs] array
    filtered_state_means : [n_timesteps, n_dim_state] array
    filtered_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
    transition_jacs_stored : [n_timesteps, n_dim_state, n_dim_state] array
        Jacobians of *f* evaluated at each filtered mean; index 0 unused.
        Required by the RTS smoother.
    """
    T = observations.shape[0]
    n_dim_state = initial_state_mean.shape[0]
    n_dim_obs = observations.shape[1]

    predicted_means = np.zeros((T, n_dim_state))
    predicted_covs = np.zeros((T, n_dim_state, n_dim_state))
    kalman_gains = np.zeros((T, n_dim_state, n_dim_obs))
    filtered_means = np.zeros((T, n_dim_state))
    filtered_covs = np.zeros((T, n_dim_state, n_dim_state))
    transition_jacs_stored = np.zeros((T, n_dim_state, n_dim_state))

    for t in range(T):
        if t == 0:
            x_pred = initial_state_mean.copy()
            P_pred = initial_state_covariance.copy()
            transition_jacs_stored[0] = np.eye(n_dim_state)
        else:
            f_t = _last_dims(transition_functions, t - 1, ndims=1)[0]
            F_t = (
                None
                if transition_jacobians is None
                else _last_dims(transition_jacobians, t - 1, ndims=1)[0]
            )
            x_pred, P_pred, Ft = _ekf_predict(
                f_t, F_t, filtered_means[t - 1], filtered_covs[t - 1],
                transition_covariance,
            )
            transition_jacs_stored[t] = Ft

        predicted_means[t] = x_pred
        predicted_covs[t] = P_pred

        g_t = _last_dims(observation_functions, t, ndims=1)[0]
        H_t = (
            None
            if observation_jacobians is None
            else _last_dims(observation_jacobians, t, ndims=1)[0]
        )
        x_filt, P_filt, K, _ = _ekf_correct(
            g_t, H_t, x_pred, P_pred, observation_covariance, observations[t]
        )
        filtered_means[t] = x_filt
        filtered_covs[t] = P_filt
        kalman_gains[t] = K

    return (
        predicted_means,
        predicted_covs,
        kalman_gains,
        filtered_means,
        filtered_covs,
        transition_jacs_stored,
    )


def _extended_kalman_smoother(
    filtered_means,
    filtered_covs,
    predicted_means,
    predicted_covs,
    transition_jacs_stored,
):
    """Apply the Rauch-Tung-Striebel smoother for the EKF.

    Parameters
    ----------
    filtered_means : [n_timesteps, n_dim_state] array
    filtered_covs : [n_timesteps, n_dim_state, n_dim_state] array
    predicted_means : [n_timesteps, n_dim_state] array
    predicted_covs : [n_timesteps, n_dim_state, n_dim_state] array
    transition_jacs_stored : [n_timesteps, n_dim_state, n_dim_state] array
        ``transition_jacs_stored[t]`` = Jacobian of *f* evaluated at
        ``filtered_means[t-1]``, used to propagate from *t-1* to *t*.

    Returns
    -------
    smoothed_means : [n_timesteps, n_dim_state] array
    smoothed_covs : [n_timesteps, n_dim_state, n_dim_state] array
    """
    T = filtered_means.shape[0]
    smoothed_means = np.zeros_like(filtered_means)
    smoothed_covs = np.zeros_like(filtered_covs)
    smoothed_means[-1] = filtered_means[-1]
    smoothed_covs[-1] = filtered_covs[-1]

    for t in reversed(range(T - 1)):
        Ft1 = transition_jacs_stored[t + 1]
        P_pred = predicted_covs[t + 1]
        smoother_gain = filtered_covs[t] @ Ft1.T @ linalg.pinv(P_pred)
        smoothed_means[t] = filtered_means[t] + smoother_gain @ (
            smoothed_means[t + 1] - predicted_means[t + 1]
        )
        smoothed_covs[t] = (
            filtered_covs[t]
            + smoother_gain @ (smoothed_covs[t + 1] - P_pred) @ smoother_gain.T
        )

    return smoothed_means, smoothed_covs


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    r"""Extended Kalman Filter and Smoother for nonlinear state-space models.

    Assumes **additive** Gaussian noise:

    .. math::

        x_0       &\sim \mathcal{N}(\mu_0, \Sigma_0)        \\
        x_{t+1}   &=    f_t(x_t) + \mathcal{N}(0, Q)        \\
        z_t       &=    g_t(x_t) + \mathcal{N}(0, R)

    The filter linearises the nonlinear functions around the current state
    estimate via first-order Taylor expansion and applies the standard
    Kalman equations on the linearised model.  The smoother (accessible via
    :meth:`smooth`) runs a Rauch-Tung-Striebel backward pass.

    Jacobians ``F_t = df_t/dx`` and ``H_t = dg_t/dx`` may be supplied
    analytically via *transition_jacobians* / *observation_jacobians* for
    better accuracy and speed.  If omitted, they are approximated by
    **central finite differences** automatically.

    Parameters
    ----------
    transition_functions : callable or array-like of callables
        State transition function(s) :math:`f_t`.  A single callable is
        reused at every time step.  Signature: ``f(x) -> x_next``.
    observation_functions : callable or array-like of callables
        Observation function(s) :math:`g_t`.  Signature: ``g(x) -> z``.
    transition_jacobians : callable or array-like of callables or None
        Jacobian(s) of :math:`f_t` w.r.t. the state, returning an
        :math:`[n, n]` array.  Signature: ``F(x) -> [n, n] array``.
        *None* (default) triggers automatic finite-difference approximation.
    observation_jacobians : callable or array-like of callables or None
        Jacobian(s) of :math:`g_t` w.r.t. the state, returning an
        :math:`[m, n]` array.  Signature: ``H(x) -> [m, n] array``.
        *None* (default) triggers automatic finite-difference approximation.
    transition_covariance : [n_dim_state, n_dim_state] array
        Process noise covariance :math:`Q`.
    observation_covariance : [n_dim_obs, n_dim_obs] array
        Observation noise covariance :math:`R`.
    initial_state_mean : [n_dim_state] array
        Mean :math:`\mu_0` of the initial state distribution.
    initial_state_covariance : [n_dim_state, n_dim_state] array
        Covariance :math:`\Sigma_0` of the initial state distribution.
    n_dim_state : int, optional
        Dimensionality of the state space.  Inferred from covariance
        parameters when possible.
    n_dim_obs : int, optional
        Dimensionality of the observation space.  Inferred from
        *observation_covariance* when possible.
    random_state : int or RandomState, optional
        Seed for :meth:`sample`.

    Examples
    --------
    Constant-velocity model with a nonlinear (quadratic) observation:

    >>> import numpy as np
    >>> from pykalman import ExtendedKalmanFilter
    >>> A = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> f = lambda x: A @ x
    >>> g = lambda x: np.array([x[0] ** 2])
    >>> ekf = ExtendedKalmanFilter(
    ...     transition_functions=f,
    ...     observation_functions=g,
    ...     transition_covariance=np.eye(2) * 0.1,
    ...     observation_covariance=np.eye(1) * 1.0,
    ...     initial_state_mean=np.array([1.0, 0.5]),
    ...     initial_state_covariance=np.eye(2),
    ...     n_dim_obs=1,
    ...     random_state=0,
    ... )
    >>> _, z = ekf.sample(50)
    >>> mu, sigma = ekf.filter(z)
    >>> mu.shape
    (50, 2)

    See Also
    --------
    AdditiveUnscentedKalmanFilter : Higher-order approximation via sigma
        points; more accurate for strongly nonlinear systems at greater
        computational cost.
    """

    def __init__(
        self,
        transition_functions=None,
        observation_functions=None,
        transition_jacobians=None,
        observation_jacobians=None,
        transition_covariance=None,
        observation_covariance=None,
        initial_state_mean=None,
        initial_state_covariance=None,
        n_dim_state=None,
        n_dim_obs=None,
        random_state=None,
    ):
        n_dim_state = _determine_dimensionality(
            [
                (transition_covariance, array2d, -2),
                (initial_state_covariance, array2d, -2),
                (initial_state_mean, array1d, -1),
            ],
            n_dim_state,
        )
        n_dim_obs = _determine_dimensionality(
            [(observation_covariance, array2d, -2)], n_dim_obs
        )

        self.transition_functions = transition_functions
        self.observation_functions = observation_functions
        self.transition_jacobians = transition_jacobians
        self.observation_jacobians = observation_jacobians
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _converters(self):
        return {
            "transition_functions": array1d,
            "observation_functions": array1d,
            "transition_covariance": array2d,
            "observation_covariance": array2d,
            "initial_state_mean": array1d,
            "initial_state_covariance": array2d,
            "n_dim_state": int,
            "n_dim_obs": int,
            "random_state": check_random_state,
        }

    def _default_parameters(self):
        return {
            "transition_functions": lambda x: x,
            "observation_functions": lambda x: x,
            "transition_covariance": np.eye(self.n_dim_state),
            "observation_covariance": np.eye(self.n_dim_obs),
            "initial_state_mean": np.zeros(self.n_dim_state),
            "initial_state_covariance": np.eye(self.n_dim_state),
            "random_state": 0,
        }

    def _initialize_parameters(self):
        """Retrieve and validate all model parameters.

        Jacobians are extracted separately because they may legitimately be
        *None* (meaning "use finite differences") and therefore cannot flow
        through :func:`preprocess_arguments`, which treats *None* as "not
        provided".
        """
        # Separate Jacobians before passing to preprocess_arguments
        trans_jac = self.transition_jacobians
        obs_jac = self.observation_jacobians

        # Remaining params go through the standard preprocessing pipeline
        args = {
            k: v
            for k, v in get_params(self).items()
            if k not in ("transition_jacobians", "observation_jacobians")
        }
        processed = preprocess_arguments(
            [args, self._default_parameters()], self._converters()
        )

        # Wrap single callables in arrays for _last_dims compatibility
        if trans_jac is not None:
            trans_jac = array1d(trans_jac)
        if obs_jac is not None:
            obs_jac = array1d(obs_jac)

        return (
            processed["transition_functions"],
            processed["observation_functions"],
            trans_jac,
            obs_jac,
            processed["transition_covariance"],
            processed["observation_covariance"],
            processed["initial_state_mean"],
            processed["initial_state_covariance"],
        )

    def _parse_observations(self, obs):
        obs = ma.atleast_2d(obs)
        if obs.shape[0] == 1 and obs.shape[1] > 1:
            obs = obs.T
        return obs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, n_timesteps, initial_state=None, random_state=None):
        """Sample a state-observation trajectory from the model.

        Parameters
        ----------
        n_timesteps : int
            Number of time steps.
        initial_state : [n_dim_state] array, optional
            Initial state.  Sampled from :math:`\\mathcal{N}(\\mu_0,\\Sigma_0)`
            when not specified.
        random_state : int or RandomState, optional
            Overrides ``self.random_state`` for this call.

        Returns
        -------
        states : [n_timesteps, n_dim_state] array
        observations : [n_timesteps, n_dim_obs] masked array
        """
        (
            transition_functions,
            observation_functions,
            _,
            _,
            transition_covariance,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        n_dim_state = transition_covariance.shape[-1]
        n_dim_obs = observation_covariance.shape[-1]

        rng = check_random_state(random_state if random_state is not None else self.random_state)

        if initial_state is None:
            initial_state = rng.multivariate_normal(
                initial_state_mean, initial_state_covariance
            )

        x = np.zeros((n_timesteps, n_dim_state))
        z = np.zeros((n_timesteps, n_dim_obs))

        for t in range(n_timesteps):
            if t == 0:
                x[0] = initial_state
            else:
                f_t = _last_dims(transition_functions, t - 1, ndims=1)[0]
                cov_q = newbyteorder(transition_covariance, "=")
                noise = rng.multivariate_normal(np.zeros(n_dim_state), cov_q)
                x[t] = np.atleast_1d(f_t(x[t - 1])) + noise

            g_t = _last_dims(observation_functions, t, ndims=1)[0]
            cov_r = newbyteorder(observation_covariance, "=")
            obs_noise = rng.multivariate_normal(np.zeros(n_dim_obs), cov_r)
            z[t] = np.atleast_1d(g_t(x[t])) + obs_noise

        return x, ma.asarray(z)

    def filter(self, Z):
        """Run the Extended Kalman Filter.

        Parameters
        ----------
        Z : [n_timesteps, n_dim_obs] array
            Observations.  Masked entries are treated as missing.

        Returns
        -------
        filtered_state_means : [n_timesteps, n_dim_state] array
        filtered_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
        """
        Z = self._parse_observations(Z)

        (
            transition_functions,
            observation_functions,
            transition_jacobians,
            observation_jacobians,
            transition_covariance,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        (_, _, _, filtered_means, filtered_covs, _) = _extended_kalman_filter(
            transition_functions,
            observation_functions,
            transition_jacobians,
            observation_jacobians,
            transition_covariance,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
            Z,
        )

        return filtered_means, filtered_covs

    def filter_update(
        self,
        filtered_state_mean,
        filtered_state_covariance,
        observation=None,
        transition_function=None,
        transition_covariance=None,
        transition_jacobian=None,
        observation_function=None,
        observation_covariance=None,
        observation_jacobian=None,
    ):
        r"""One-step update for online (streaming) filtering.

        Given the posterior at time *t*, incorporate a new observation at
        time *t+1* to produce the posterior at *t+1*.

        Parameters
        ----------
        filtered_state_mean : [n_dim_state] array
            :math:`\hat{x}_{t|t}`
        filtered_state_covariance : [n_dim_state, n_dim_state] array
            :math:`P_{t|t}`
        observation : [n_dim_obs] array or None
            Observation at *t+1*.  *None* or a fully masked array → missing.
        transition_function : callable or None
            Override ``self.transition_functions`` for this step.
        transition_covariance : [n_dim_state, n_dim_state] array or None
            Override ``self.transition_covariance`` for this step.
        transition_jacobian : callable or None
            Override ``self.transition_jacobians`` for this step.
        observation_function : callable or None
            Override ``self.observation_functions`` for this step.
        observation_covariance : [n_dim_obs, n_dim_obs] array or None
            Override ``self.observation_covariance`` for this step.
        observation_jacobian : callable or None
            Override ``self.observation_jacobians`` for this step.

        Returns
        -------
        next_filtered_state_mean : [n_dim_state] array
        next_filtered_state_covariance : [n_dim_state, n_dim_state] array
        """
        (
            transition_functions,
            observation_functions,
            transition_jacobians,
            observation_jacobians,
            transition_cov,
            observation_cov,
            _,
            _,
        ) = self._initialize_parameters()

        def _pick(override, stored_arr):
            if override is not None:
                return override
            return _last_dims(stored_arr, 0, ndims=1)[0]

        def _pick_jac(override, stored_arr):
            if override is not None:
                return override
            if stored_arr is None:
                return None
            return _last_dims(stored_arr, 0, ndims=1)[0]

        f = _pick(transition_function, transition_functions)
        g = _pick(observation_function, observation_functions)
        F = _pick_jac(transition_jacobian, transition_jacobians)
        H = _pick_jac(observation_jacobian, observation_jacobians)
        Q = array2d(transition_covariance) if transition_covariance is not None else transition_cov
        R = array2d(observation_covariance) if observation_covariance is not None else observation_cov

        if observation is None:
            observation = ma.array(
                np.zeros(self.n_dim_obs),
                mask=np.ones(self.n_dim_obs, dtype=bool),
            )

        x_pred, P_pred, _ = _ekf_predict(f, F, filtered_state_mean, filtered_state_covariance, Q)
        x_filt, P_filt, _, _ = _ekf_correct(g, H, x_pred, P_pred, R, observation)

        return x_filt, P_filt

    def smooth(self, Z):
        """Run the Extended Kalman Filter followed by the RTS smoother.

        Parameters
        ----------
        Z : [n_timesteps, n_dim_obs] array
            Observations.  Masked entries are treated as missing.

        Returns
        -------
        smoothed_state_means : [n_timesteps, n_dim_state] array
        smoothed_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
        """
        Z = self._parse_observations(Z)

        (
            transition_functions,
            observation_functions,
            transition_jacobians,
            observation_jacobians,
            transition_covariance,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        (
            predicted_means,
            predicted_covs,
            _,
            filtered_means,
            filtered_covs,
            transition_jacs_stored,
        ) = _extended_kalman_filter(
            transition_functions,
            observation_functions,
            transition_jacobians,
            observation_jacobians,
            transition_covariance,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
            Z,
        )

        smoothed_means, smoothed_covs = _extended_kalman_smoother(
            filtered_means,
            filtered_covs,
            predicted_means,
            predicted_covs,
            transition_jacs_stored,
        )

        return smoothed_means, smoothed_covs
