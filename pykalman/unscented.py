'''
=========================================
Inference for Non-Linear Gaussian Systems
=========================================

This module contains the Unscented Kalman Filter (Wan, van der Merwe 2000)
for state estimation in systems with non-Gaussian noise and non-linear dynamics
'''
import numpy as np
from numpy import ma
from scipy import linalg

from .utils import array1d, array2d, check_random_state

from .standard import _last_dims


def _unscented_transform2(
        mean, covariance2, f, noise_covariance2=None,
        alpha=1e-3, beta=2.0, kappa=0.0
    ):
    '''
    Parameters
    ----------
    mean: [n_dim] array
    covariance2: [n_dim, n_dim] array
        square root of covariance matrix
    f : n_dim -> n_dim function
    noise_covariance2 : [n_dim, n_dim] array
        square of covariance matrix of additive noise

    Returns
    -------
    mean: [n_dim] array
        weighted mean of sigma points propagated through f
    covariance2: [n_dim, n_dim] array
        square root of weighted covariance of sigma points propagated through f
    '''
    n_dim = len(mean)
    if noise_covariance2 is None:
      noise_covariance2 = np.zeros((n_dim, n_dim))
    mean = np.asarray(mean, dtype=float)

    # calculate scaling factor for all off-center points
    lamda = (alpha * alpha) * (n_dim + kappa) - n_dim
    c = n_dim + lamda

    # calculate weights
    weights_mean = np.ones(2 * n_dim + 1)
    weights_mean[0] = lamda / c
    weights_mean[1:] = 0.5 / c
    weights_cov = np.copy(weights_mean)
    weights_cov[0] = lamda / c + (1 - alpha * alpha + beta)

    # make sigma points
    points = np.tile(mean[:, np.newaxis], (1, 2 * n_dim + 1))
    points[:, 1:n_dim+1] = mean[:, np.newaxis] + sigma2 * np.sqrt(c)
    points[:, n_dim+1:]  = mean[:, np.newaxis] - sigma2 * np.sqrt(c)

    # propagate through function
    transformed_points = [f(points[:, i]) for i in range(2 * n_dim + 1)]
    transformed_points = np.vstack(transformed_points)

    # calculate new mean
    transformed_mean = transformed_points.T.dot(weights_mean)

    # calculate new square root covariance
    transformed_covariance2 = qr(np.hstack([
      np.sqrt(weights_cov[1]) * (transformed_points[:, 1:] - transformed_mean[:, np.newaxis]),
      noise_covariance2
    ]))
    transformed_covariance2 = cholupdate(
        transformed_covariance2,
        (transformed_points[:, 0] - transformed_mean[:, np.newaxis]).T,
        weights_cov[0]
    )
    return (transformed_mean, transformed_covariance2, transformed_points)


def cholupdate(A2, x, weight):
  '''Calculate chol(A + w x x')

  Parameters
  ----------
  A2 : [n_dim, n_dim] array
      A = A2.T.dot(A2) for A positive definite, symmetric
  x : [n_dim] or [n_vec, n_dim] array
      vector(s) to be used for x.  If x has 2 dimensions, then each row will be
      added in turn.
  weight : float
      square of weight to be multiplied to each x x'. If negative, will use
      sign(weight) * sqrt(abs(weight)) instead of sqrt(weight).

  Returns
  -------
  A2 : [n_dim, n_dim array]
      cholesky decomposition of updated matrix
  '''
  if len(x.shape) == 1:
      x = x[np.newaxis,:]
  n_vec, n_dim = x.shape

  M = A2.T.dot(A2)
  for i in range(n_vec):
      M += np.sign(weight) * np.sqrt(np.abs(weight)) * np.outer(x[i], x[i])

  return linalg.cholesky(M)


def qr(A):
    '''Get square upper triangular matrix of QR decomposition of matrix A'''
    N, L = A.shape
    if not N >= L:
        raise ValueError("Number of columns must exceed number of rows")
    Q, R = linalg.qr(A)
    return R[:L, :L]


def _unscented_moments(points, weights_mu, weights_sigma, sigma2_noise=None):
    '''Calculate the weighted mean and covariance of `points`

    Parameters
    ----------
    points : [2 * n_dim_state + 1, n_dim_state] array
        array where each row is a sigma point
    weights_mu : [2 * n_dim_state + 1] array
        weights used to calculate the mean
    weights_sigma : [2 * n_dim_state + 1] array
        weights used to calcualte the covariance
    sigma2_noise : [n_dim_state, n_dim_state] array
        square root of additive noise covariance matrix

    Returns
    -------
    mu : [n_dim_state] array
        approximate mean
    sigma2 : [n_dim_state, n_dim_state] array
        R s.t. R' R = approximate covariance
    '''
    mu = points.T.dot(weights_mu)

    # make points to perform QR factorization on. each column is one data point
    qr_points = [
        np.sign(weights_sigma)[np.newaxis, :]
        * np.sqrt(np.abs(weights_sigma))[np.newaxis, :]
        * (points.T - mu[:, np.newaxis])
    ]
    if sigma2_noise is not None:
        qr_points.append(sigma2_noise)
    sigma2 = qr(np.hstack(qr_points).T)
    #sigma2 = cholupdate(sigma2, points[0] - mu, weights_sigma[0])
    return (mu.ravel(), sigma2)


def _sigma_points(mu, sigma2, alpha=1e-3, beta=2.0, kappa=0.0):
    '''Calculate "sigma points" used in Unscented Kalman Filter

    Parameters
    ----------
    mu : [n_dim] array
        Mean of multivariate normal distribution
    sigma2 : [n_dim, n_dim] array
        R s.t. R' R = Covariance of multivariate normal
    alpha : float
        Spread of the sigma points. Typically 1e-3.
    beta : float
        Used to "incorporate prior knowledge of the distribution of the state".
        2 is optimal is the state is normally distributed.
    kappa : float
        a parameter which means ????

    Returns
    -------
    points : [2*n_dim+1, n_dim] array
        each row is a sigma point of the UKF
    weights_mean : [2*n_dim+1] array
        weights for calculating the empirical mean
    weights_cov : [n*n_dim+1] array
        weights for calculating the empirical covariance
    '''
    n_dim = len(mu)
    mu = array2d(mu, dtype=float)

    # just because I saw it in the MATLAB implementation
    sigma2 = sigma2.T

    # Calculate scaling factor for all off-center points
    lamda = (alpha * alpha) * (n_dim + kappa) - n_dim
    c = n_dim + lamda

    # calculate the sigma points; that is,
    #   mu
    #   mu + each column of sigma2 * sqrt(c)
    #   mu - each column of sigma2 * sqrt(c)
    # Each column of points is one of these.
    points = np.tile(mu.T, (1, 2 * n_dim + 1))
    points[:, 1:(n_dim + 1)] += sigma2 * np.sqrt(c)
    points[:, (n_dim + 1):] -= sigma2 * np.sqrt(c)

    # Calculate weights
    weights_mean = np.ones(2 * n_dim + 1)
    weights_mean[0] = lamda / c
    weights_mean[1:] = 0.5 / c
    weights_cov = np.copy(weights_mean)
    weights_cov[0] = lamda / c + (1 - alpha * alpha + beta)

    return (points.T, weights_mean, weights_cov)


def _unscented_transform(f, points, weights_mean, weights_cov,
                         points_noise=None, sigma2_noise=None):
    '''Apply the Unscented Transform.

    Parameters
    ==========
    f : [n_dim_1, n_dim_3] -> [n_dim_2] function
        function to apply pass all points through
    points : [n_points, n_dim_1] array
        points representing state to pass through `f`
    weights_mean : [n_points] array
        weights used to calculate the empirical mean
    weights_cov : [n_points] array
        weights used to calculate empirical covariance
    points_noise : [n_points, n_dim_3] array
        points representing noise to pass through `f`, if any.
    sigma2_noise : [n_dim_2, n_dim_2] array
        square root of covariance matrix for additive noise

    Returns
    =======
    points_pred : [n_points, n_dim_2] array
        points passed through f
    mu_pred : [n_dim_2] array
        empirical mean
    sigma2_pred : [n_dim_2, n_dim_2] array
        R s.t. R' R = empirical covariance
    '''
    n_points = points.shape[0]

    # propagate points through f.  Each column is a sample point
    if points_noise is None:
        points_pred = [f(points[i]) for i in range(n_points)]
    else:
        points_pred = [f(points[i], points_noise[i]) for i in range(n_points)]

    # make each row a predicted point
    points_pred = np.vstack(points_pred)

    # calculate approximate mean, covariance
    (mu_pred, sigma2_pred) = _unscented_moments(
        points_pred, weights_mean, weights_cov, sigma2_noise
    )

    return (points_pred, mu_pred, sigma2_pred)


def _unscented_correct(cross_sigma, mu_pred, sigma2_pred, obs_mu_pred,
                       obs_sigma2_pred, z):
    '''Correct predicted state estimates with an observation

    Parameters
    ----------
    cross_sigma : [n_dim_state, n_dim_obs] array
        cross-covariance between the state at time t given all observations
        from timesteps [0, t-1] and the observation at time t
    mu_pred : [n_dim_state] array
        mean of state at time t given observations from timesteps [0, t-1]
    sigma2_pred : [n_dim_state, n_dim_state] array
        square root of covariance of state at time t given observations from
        timesteps [0, t-1]
    obs_mu_pred : [n_dim_obs] array
        mean of observation at time t given observations from times [0, t-1]
    obs_sigma2_pred : [n_dim_obs] array
        square root of covariance of observation at time t given observations
        from times [0, t-1]
    z : [n_dim_obs] array
        observation at time t

    Returns
    -------
    mu_filt : [n_dim_state] array
        mean of state at time t given observations from time steps [0, t]
    sigma2_filt : [n_dim_state, n_dim_state] array
        square root of covariance of state at time t given observations from
        time steps [0, t]
    '''
    n_dim_state = len(mu_pred)
    n_dim_obs = len(obs_mu_pred)

    if not np.any(ma.getmask(z)):
        # calculate Kalman gain
        K = cross_sigma.dot(linalg.pinv(obs_sigma2_pred.T.dot(obs_sigma2_pred)))
        #K = linalg.lstsq(cross_sigma.T, obs_sigma2_pred.T)[0]
        #K = linalg.lstsq(K.T, obs_sigma2_pred)[0]

        # correct mu, sigma
        mu_filt = mu_pred + K.dot(z - obs_mu_pred)
        U = K.dot(obs_sigma2_pred)
        sigma2_filt = cholupdate(sigma2_pred, U.T, -1.0)
    else:
        # no corrections to be made
        mu_filt = mu_pred
        sigma2_filt = sigma2_pred
    return (mu_filt, sigma2_filt)


def _augment(means, covariances):
    '''Calculate augmented mean and covariance matrix

    Parameters
    ----------
    means : list of 1D arrays
        means of multiple independent multivariate normal random variables
    covariances : list of 2D square arrays
        covariances of corresponding means

    Returns
    -------
    mu_aug : 1D array
        all means, concatenated together
    sigma_aug : 2D array
        block diagonal covariance matrix constructed from `covariances`
    '''
    mu_aug = np.concatenate(means)
    sigma_aug = linalg.block_diag(*covariances)
    return (mu_aug, sigma_aug)


def _unaugment_points(points, dims):
    '''Extract unaugmented portion of augmented sigma points

    Parameters
    ----------
    points : 2D array
        array where each row is an augmented point
    dims : array of int
        size of each component in it appears in `points

    Returns
    -------
    result : list of 2D arrays
        A list of arrays, each representing the part of `points` corresponding
        to each component specified in `dims`
    '''
    result = []
    start = 0
    for d in dims:
        stop = start + d
        result.append(points[:, start:stop])
        start = stop
    return result


def _augmented_unscented_filter(mu_0, sigma_0, f, g, Q, R, Z):
    '''Apply the Unscented Kalman Filter with arbitrary noise

    Parameters
    ----------
    mu_0 : [n_dim_state] array
        mean of initial state distribution
    sigma_0 : [n_dim_state, n_dim_state] array
        covariance of initial state distribution
    f : function or [T-1] array of functions
        state transition function(s). Takes in an the current state and the
        process noise and outputs the next state.
    g : function or [T] array of functions
        observation function(s). Takes in the current state and outputs the
        current observation.
    Q : [n_dim_state, n_dim_state] array
        transition covariance matrix
    R : [n_dim_state, n_dim_state] array
        observation covariance matrix

    Returns
    -------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of state at time t given observations from times [0,
        t]
    sigma_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of state at time t given observations from
        times [0, t]
    '''
    # extract size of key components
    T = Z.shape[0]
    n_dim_state = Q.shape[-1]
    n_dim_obs = R.shape[-1]

    # construct container for results
    mu_filt = np.zeros((T, n_dim_state))
    sigma_filt = np.zeros((T, n_dim_state, n_dim_state))

    for t in range(T):
        # Calculate sigma points for augmented state:
        #   [actual state, transition noise, observation noise]
        if t == 0:
            mu, sigma = mu_0, sigma_0
        else:
            mu, sigma = mu_filt[t - 1], sigma_filt[t - 1]

        (mu_aug, sigma_aug) = _augment(
            [mu, np.zeros(n_dim_state), np.zeros(n_dim_obs)],
            [sigma, Q, R]
        )
        (points_aug, weights_mu, weights_sigma) = (
            _sigma_points(mu_aug, sigma_aug)
        )
        (points_state, points_trans, points_obs) = (
            _unaugment_points(points_aug,
                              [n_dim_state, n_dim_state, n_dim_obs])
        )

        # Calculate E[x_t | z_{0:t-1}], Var(x_t | z_{0:t-1}) and sigma points
        # for P(x_t | z_{0:t-1})
        if t == 0:
            points_pred = points_state
            (mu_pred, sigma_pred) = (
                _unscented_moments(points_pred, weights_mu, weights_sigma)
            )
        else:
            f_t1 = _last_dims(f, t - 1, ndims=1)[0]
            (points_pred, mu_pred, sigma_pred) = (
                _unscented_transform(f_t1, points_state, weights_mu,
                                     weights_sigma, points_trans)
            )

        # Calculate E[z_t | z_{0:t-1}], Var(z_t | z_{0:t-1})
        g_t = _last_dims(g, t, ndims=1)[0]
        (obs_points_pred, obs_mu_pred, obs_sigma_pred) = (
            _unscented_transform(g_t, points_pred, weights_mu, weights_sigma,
                                 points_obs)
        )

        # Calculate Cov(x_t, z_t | z_{0:t-1})
        sigma_pair = (
            ((points_pred - mu_pred).T)
            .dot(np.diag(weights_sigma))
            .dot(obs_points_pred - obs_mu_pred)
        )

        # Calculate E[x_t | z_{0:t}], Var(x_t | z_{0:t})
        (mu_filt[t], sigma_filt[t]) = (
            _unscented_correct(sigma_pair, mu_pred, sigma_pred, obs_mu_pred,
                               obs_sigma_pred, Z[t])
        )

    return (mu_filt, sigma_filt)


def _augmented_unscented_smoother(mu_filt, sigma_filt, f, Q):
    '''Apply the Unscented Kalman Smoother with arbitrary noise

    Parameters
    ----------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of state at time t given observations from times
        [0, t]
    sigma_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of state at time t given observations from
        times [0, t]
    f : function or [T-1] array of functions
        state transition function(s). Takes in an the current state and the
        process noise and outputs the next state.
    Q : [n_dim_state, n_dim_state] array
        transition covariance matrix

    Returns
    -------
    mu_smooth : [T, n_dim_state] array
        mu_smooth[t] = mean of state at time t given observations from times
        [0, T-1]
    sigma_smooth : [T, n_dim_state, n_dim_state] array
        sigma_smooth[t] = covariance of state at time t given observations from
        times [0, T-1]
    '''
    # extract size of key parts of problem
    T, n_dim_state = mu_filt.shape

    # instantiate containers for results
    mu_smooth = np.zeros(mu_filt.shape)
    sigma_smooth = np.zeros(sigma_filt.shape)
    mu_smooth[-1], sigma_smooth[-1] = mu_filt[-1], sigma_filt[-1]

    for t in reversed(range(T - 1)):
        # get sigma points for [state, transition noise]
        mu = mu_filt[t]
        sigma = sigma_filt[t]

        (mu_aug, sigma_aug) = _augment(
            [mu, np.zeros(n_dim_state)],
            [sigma, Q]
        )
        (points_aug, weights_mu, weights_sigma) = (
            _sigma_points(mu_aug, sigma_aug)
        )
        (points_state, points_trans) = (
            _unaugment_points(points_aug, [n_dim_state, n_dim_state])
        )

        # compute E[x_{t+1} | z_{0:t}], Var(x_{t+1} | z_{0:t})
        f_t = _last_dims(f, t, ndims=1)[0]
        (points_pred, mu_pred, sigma_pred) = (
            _unscented_transform(f_t, points_state, weights_mu, weights_sigma,
                                 points_trans)
        )

        # Calculate Cov(x_{t+1}, x_t | z_{0:t-1})
        sigma_pair = (
            (points_pred - mu_pred).T
            .dot(np.diag(weights_sigma))
            .dot(points_state - mu).T
        )

        # compute smoothed mean, covariance
        smoother_gain = sigma_pair.dot(linalg.pinv(sigma_pred))
        mu_smooth[t] = (
            mu_filt[t]
            + smoother_gain
              .dot(mu_smooth[t + 1] - mu_pred)
        )
        sigma_smooth[t] = (
            sigma_filt[t]
            + smoother_gain
              .dot(sigma_smooth[t + 1] - sigma_pred)
              .dot(smoother_gain.T)
        )

    return (mu_smooth, sigma_smooth)


class UnscentedKalmanFilter():
    r'''Implements the General (aka Augmented) Unscented Kalman Filter governed
    by the following equations,

    .. math::

        v_t       &\sim \text{Normal}(0, Q)     \\
        w_t       &\sim \text{Normal}(0, R)     \\
        x_{t+1}   &= f_t(x_t, v_t)              \\
        z_{t}     &= g_t(x_t, w_t)

    Notice that although the input noise to the state transition equation and
    the observation equation are both normally distributed, any non-linear
    transformation may be applied afterwards.  This allows for greater
    generality, but at the expense of computational complexity.  The complexity
    of :class:`UnscentedKalmanFilter.filter()` is :math:`O(T(2n+m)^3)`
    where :math:`T` is the number of time steps, :math:`n` is the size of the
    state space, and :math:`m` is the size of the observation space.

    If your noise is simply additive, consider using the
    :class:`AdditiveUnscentedKalmanFilter`

    Parameters
    ----------
    f : function or [T-1] array of functions
        f[t] is a function of the state and the transition noise at time t and
        produces the state at time t+1
    g : function or [T] array of functions
        g[t] is a function of the state and the observation noise at time t and
        produces the observation at time t.
    Q : [n_dim_state, n_dim_state] array
        transition noise covariance matrix
    R : [n_dim_obs, n_dim_obs] array
        observation noise covariance matrix
    mu_0 : [n_dim_state] array
        mean of initial state distribution
    sigma_0 : [n_dim_state, n_dim_state] array
        covariance of initial state distribution
    random_state : optional, int or RandomState
        seed for random sample generation
    '''
    def __init__(self, f, g, Q, R, mu_0, sigma_0, random_state=None):
        self.f = array1d(f)
        self.g = array1d(g)
        self.Q = array2d(Q)
        self.R = array2d(R)
        self.mu_0 = array1d(mu_0)
        self.sigma_0 = array2d(sigma_0)
        self.random_state = random_state

    def sample(self, T, x_0=None):
        '''Sample from model defined by the Unscented Kalman Filter

        Parameters
        ----------
        T : int
            number of time steps
        x_0 : optional, [n_dim_state] array
            initial state.  If unspecified, will be sampled from initial state
            distribution.
        '''
        n_dim_state = self.Q.shape[-1]
        n_dim_obs = self.R.shape[-1]

        # logic for instantiating rng
        rng = check_random_state(self.random_state)

        # logic for selecting initial state
        if x_0 is None:
            x_0 = rng.multivariate_normal(self.mu_0, self.sigma_0)

        # logic for generating samples
        x = np.zeros((T, n_dim_state))
        z = np.zeros((T, n_dim_obs))
        for t in range(T):
            if t == 0:
                x[0] = x_0
            else:
                f_t1 = _last_dims(self.f, t - 1, ndims=1)[0]
                Q_t1 = self.Q
                e_t1 = rng.multivariate_normal(np.zeros(n_dim_state),
                                               Q_t1.newbyteorder('='))
                x[t] = f_t1(x[t - 1], e_t1)

            g_t = _last_dims(self.g, t, ndims=1)[0]
            R_t = self.R
            e_t2 = rng.multivariate_normal(np.zeros(n_dim_obs),
                                           R_t.newbyteorder('='))
            z[t] = g_t(x[t], e_t2)

        return (x, ma.asarray(z))

    def filter(self, Z):
        '''Run Unscented Kalman Filter

        Parameters
        ----------
        Z : [T, n_dim_state] array
            Z[t] = observation at time t.  If Z is a masked array and any of
            Z[t]'s elements are masked, the observation is assumed missing and
            ignored.

        Returns
        -------
        mu_filt : [T, n_dim_state] array
            mu_filt[t] = mean of state distribution at time t given
            observations from times [0, t]
        sigma_filt : [T, n_dim_state, n_dim_state] array
            sigma_filt[t] = covariance of state distribution at time t given
            observations from times [0, t]
        '''
        Z = ma.asarray(Z)

        (mu_filt, sigma_filt) = _augmented_unscented_filter(
            self.mu_0, self.sigma_0, self.f,
            self.g, self.Q, self.R, Z
        )

        return (mu_filt, sigma_filt)

    def smooth(self, Z):
        '''Run Unscented Kalman Smoother

        Parameters
        ----------
        Z : [T, n_dim_state] array
            Z[t] = observation at time t.  If Z is a masked array and any of
            Z[t]'s elements are masked, the observation is assumed missing and
            ignored.

        Returns
        -------
        mu_smooth : [T, n_dim_state] array
            mu_filt[t] = mean of state distribution at time t given
            observations from times [0, T-1]
        sigma_smooth : [T, n_dim_state, n_dim_state] array
            sigma_filt[t] = covariance of state distribution at time t given
            observations from times [0, T-1]
        '''
        Z = ma.asarray(Z)

        (mu_filt, sigma_filt) = self.filter(Z)
        (mu_smooth, sigma_smooth) = _augmented_unscented_smoother(
            mu_filt, sigma_filt, self.f, self.Q
        )

        return (mu_smooth, sigma_smooth)


def _additive_unscented_filter(mu_0, sigma_0, f, g, Q, R, Z):
    '''Apply the Unscented Kalman Filter with additive noise

    Parameters
    ----------
    mu_0 : [n_dim_state] array
        mean of initial state distribution
    sigma_0 : [n_dim_state, n_dim_state] array
        covariance of initial state distribution
    f : function or [T-1] array of functions
        state transition function(s). Takes in an the current state and outputs
        the next.
    g : function or [T] array of functions
        observation function(s). Takes in the current state and outputs the
        current observation.
    Q : [n_dim_state, n_dim_state] array
        transition covariance matrix
    R : [n_dim_state, n_dim_state] array
        observation covariance matrix

    Returns
    -------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of state at time t given observations from times [0,
        t]
    sigma2_filt : [T, n_dim_state, n_dim_state] array
        sigma2_filt[t] = square root of the covariance of state at time t given
        observations from times [0, t]
    '''
    # extract size of key components
    T = Z.shape[0]
    n_dim_state = Q.shape[-1]
    n_dim_obs = R.shape[-1]

    # construct container for results
    mu_filt = np.zeros((T, n_dim_state))
    sigma2_filt = np.zeros((T, n_dim_state, n_dim_state))
    Q2 = linalg.cholesky(Q)
    R2 = linalg.cholesky(R)

    for t in range(T):
        # Calculate sigma points for P(x_{t-1} | z_{0:t-1})
        if t == 0:
            mu, sigma2 = mu_0, linalg.cholesky(sigma_0)
        else:
            mu, sigma2 = mu_filt[t - 1], sigma2_filt[t - 1]

        (points_state, weights_mu, weights_sigma) = _sigma_points(mu, sigma2)

        # Calculate E[x_t | z_{0:t-1}], Var(x_t | z_{0:t-1})
        if t == 0:
            points_pred = points_state
            (mu_pred, sigma2_pred) = (
                _unscented_moments(points_pred, weights_mu, weights_sigma)
            )
        else:
            f_t1 = _last_dims(f, t - 1, ndims=1)[0]

            (_, mu_pred, sigma2_pred) = (
                _unscented_transform(f_t1, points_state,
                                     weights_mu, weights_sigma,
                                     sigma2_noise=Q2)
            )
            points_pred = _sigma_points(mu_pred, sigma2_pred)[0]

        # Calculate E[z_t | z_{0:t-1}], Var(z_t | z_{0:t-1})
        g_t = _last_dims(g, t, ndims=1)[0]
        (obs_points_pred, obs_mu_pred, obs_sigma2_pred) = (
            _unscented_transform(g_t, points_pred,
                                 weights_mu, weights_sigma,
                                 sigma2_noise=R2)
        )

        # Calculate Cov(x_t, z_t | z_{0:t-1})
        sigma_pair = (
            (points_pred - mu_pred).T
            .dot(np.diag(weights_sigma))
            .dot(obs_points_pred - obs_mu_pred)
        )

        # Calculate E[x_t | z_{0:t}], Var(x_t | z_{0:t})
        (mu_filt[t], sigma2_filt[t]) = (
            _unscented_correct(sigma_pair,
                               mu_pred, sigma2_pred,
                               obs_mu_pred, obs_sigma2_pred,
                               Z[t])
        )

    return (mu_filt, sigma2_filt)


def _additive_unscented_smoother(mu_filt, sigma2_filt, f, Q):
    '''Apply the Unscented Kalman Filter assuming additiven noise

    Parameters
    ----------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of state at time t given observations from times
        [0, t]
    sigma_2filt : [T, n_dim_state, n_dim_state] array
        sigma2_filt[t] = square root of the covariance of state at time t given
        observations from times [0, t]
    f : function or [T-1] array of functions
        state transition function(s). Takes in an the current state and outputs
        the next.
    Q : [n_dim_state, n_dim_state] array
        transition covariance matrix

    Returns
    -------
    mu_smooth : [T, n_dim_state] array
        mu_smooth[t] = mean of state at time t given observations from times
        [0, T-1]
    sigma2_smooth : [T, n_dim_state, n_dim_state] array
        sigma2_smooth[t] = square root of the covariance of state at time t
        given observations from times [0, T-1]
    '''
    # extract size of key parts of problem
    T, n_dim_state = mu_filt.shape

    # instantiate containers for results
    mu_smooth = np.zeros(mu_filt.shape)
    sigma2_smooth = np.zeros(sigma2_filt.shape)
    mu_smooth[-1], sigma2_smooth[-1] = mu_filt[-1], sigma2_filt[-1]
    Q2 = linalg.cholesky(Q)

    for t in reversed(range(T - 1)):
        # get sigma points for state
        mu = mu_filt[t]
        sigma2 = sigma2_filt[t]

        (points_state, weights_mu, weights_sigma) = (
            _sigma_points(mu, sigma2)
        )

        # compute E[x_{t+1} | z_{0:t}], Var(x_{t+1} | z_{0:t})
        f_t = _last_dims(f, t, ndims=1)[0]
        (points_pred, mu_pred, sigma2_pred) = (
            _unscented_transform(f_t, points_state, weights_mu,
                                 weights_sigma, sigma2_noise=Q2)
        )

        # Calculate Cov(x_{t+1}, x_t | z_{0:t-1})
        sigma_pair = (
            (points_pred - mu_pred).T
            .dot(np.diag(weights_sigma))
            .dot(points_state - mu).T
        )

        # compute smoothed mean, covariance
        smoother_gain = (
            sigma_pair.dot(linalg.pinv(sigma2_pred.T.dot(sigma2_pred)))
        )
        mu_smooth[t] = (
            mu_filt[t]
            + smoother_gain
              .dot(mu_smooth[t + 1] - mu_pred)
        )
        U = cholupdate(sigma2_pred, sigma2_smooth[t + 1], -1.0)
        sigma2_smooth[t] = (
            cholupdate(sigma2_filt[t], smoother_gain.dot(U.T).T, -1.0)
        )

    return (mu_smooth, sigma2_smooth)


class AdditiveUnscentedKalmanFilter():
    r'''Implements the Unscented Kalman Filter with additive noise.
    Observations are assumed to be generated from the following process,

    .. math::

        v_t       &\sim \text{Normal}(0, Q)     \\
        w_t       &\sim \text{Normal}(0, R)     \\
        x_{t+1}   &= f_t(x_t) + v_t             \\
        z_{t}     &= g_t(x_t) + w_t

    While less general the general-noise Unscented Kalman Filter, the Additive
    version is more computationally efficient with complexity :math:`O(Tn^3)`
    where :math:`T` is the number of time steps and :math:`n` is the size of
    the state space.

    Parameters
    ----------
    f : function or [T-1] array of functions
        f[t] is a function of the state at time t and produces the state at
        time t+1
    g : function or [T] array of functions
        g[t] is a function of the state at time t and produces the observation
        at time t
    Q : [n_dim_state, n_dim_state] array
        transition noise covariance matrix
    R : [n_dim_obs, n_dim_obs] array
        observation noise covariance matrix
    mu_0 : [n_dim_state] array
        mean of initial state distribution
    sigma_0 : [n_dim_state, n_dim_state] array
        covariance of initial state distribution
    random_state : optional, int or RandomState
        seed for random sample generation
    '''
    def __init__(self, f, g, Q, R, mu_0, sigma_0, random_state=None):
        self.f = array1d(f)
        self.g = array1d(g)
        self.Q = array2d(Q)
        self.R = array2d(R)
        self.mu_0 = array1d(mu_0)
        self.sigma_0 = array2d(sigma_0)
        self.random_state = random_state

    def sample(self, T, x_0=None):
        '''Sample from model defined by the Unscented Kalman Filter

        Parameters
        ----------
        T : int
            number of time steps
        x_0 : optional, [n_dim_state] array
            initial state.  If unspecified, will be sampled from initial state
            distribution.
        '''
        n_dim_state = self.Q.shape[-1]
        n_dim_obs = self.R.shape[-1]

        # logic for instantiating rng
        rng = check_random_state(self.random_state)

        # logic for selecting initial state
        if x_0 is None:
            x_0 = rng.multivariate_normal(self.mu_0, self.sigma_0)

        # logic for generating samples
        x = np.zeros((T, n_dim_state))
        z = np.zeros((T, n_dim_obs))
        for t in range(T):
            if t == 0:
                x[0] = x_0
            else:
                f_t1 = _last_dims(self.f, t - 1, ndims=1)[0]
                Q_t1 = self.Q
                e_t1 = rng.multivariate_normal(np.zeros(n_dim_state),
                                               Q_t1.newbyteorder('='))
                x[t] = f_t1(x[t - 1]) + e_t1

            g_t = _last_dims(self.g, t, ndims=1)[0]
            R_t = self.R
            e_t2 = rng.multivariate_normal(np.zeros(n_dim_obs),
                                           R_t.newbyteorder('='))
            z[t] = g_t(x[t]) + e_t2

        return (x, ma.asarray(z))

    def filter(self, Z):
        '''Run Unscented Kalman Filter

        Parameters
        ----------
        Z : [T, n_dim_state] array
            Z[t] = observation at time t.  If Z is a masked array and any of
            Z[t]'s elements are masked, the observation is assumed missing and
            ignored.

        Returns
        -------
        mu_filt : [T, n_dim_state] array
            mu_filt[t] = mean of state distribution at time t given
            observations from times [0, t]
        sigma_filt : [T, n_dim_state, n_dim_state] array
            sigma_filt[t] = covariance of state distribution at time t given
            observations from times [0, t]
        '''
        Z = ma.asarray(Z)
        T = Z.shape[0]

        # run square root filter
        (mu_filt, sigma2_filt) = _additive_unscented_filter(
            self.mu_0, self.sigma_0, self.f,
            self.g, self.Q, self.R, Z
        )

        # reconstruct covariance matrices
        sigma_filt = np.zeros(sigma2_filt.shape)
        for t in range(T):
            sigma_filt[t] = sigma2_filt[t].T.dot(sigma2_filt[t])

        return (mu_filt, sigma_filt)

    def smooth(self, Z):
        '''Run Unscented Kalman Smoother

        Parameters
        ----------
        Z : [T, n_dim_state] array
            Z[t] = observation at time t.  If Z is a masked array and any of
            Z[t]'s elements are masked, the observation is assumed missing and
            ignored.

        Returns
        -------
        mu_smooth : [T, n_dim_state] array
            mu_filt[t] = mean of state distribution at time t given
            observations from times [0, T-1]
        sigma_smooth : [T, n_dim_state, n_dim_state] array
            sigma_filt[t] = covariance of state distribution at time t given
            observations from times [0, T-1]
        '''
        Z = ma.asarray(Z)
        T = Z.shape[0]

        # run filter, then smoother
        (mu_filt, sigma2_filt) = _additive_unscented_filter(
            self.mu_0, self.sigma_0, self.f,
            self.g, self.Q, self.R, Z
        )
        (mu_smooth, sigma2_smooth) = _additive_unscented_smoother(
            mu_filt, sigma2_filt, self.f, self.Q
        )

        # reconstruction covariance matrices
        sigma_smooth = np.zeros(sigma2_smooth.shape)
        for t in range(T):
            sigma_smooth[t] = sigma2_smooth[t].T.dot(sigma2_smooth[t])

        return (mu_smooth, sigma_smooth)
