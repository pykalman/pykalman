'''
=========================================
Inference for Non-Linear Gaussian Systems
=========================================

This module contains "Square Root" implementations to the Unscented Kalman
Filter.  Square Root implementations typically propagate the mean and Cholesky
factorization of the covariance matrix in order to prevent numerical error.
When possible, Square Root implementations should be preferred to their
standard counterparts.

References
----------

* Terejanu, G.A. Towards a Decision-Centric Framework for Uncertainty
  Propagation and Data Assimilation. 2010.
* Van Der Merwe, R. and Wan, E.A. The Square-Root Unscented Kalman Filter for
  State and Parameter-Estimation. 2001.
'''
import numpy as np
from numpy import ma
from scipy import linalg

from ..utils import array1d, array2d, check_random_state

from ..standard import _last_dims
from ..unscented import AdditiveUnscentedKalmanFilter as AUKF


def cholupdate(A2, X, weight):
  '''Calculate chol(A + w x x')

  Parameters
  ----------
  A2 : [n_dim, n_dim] array
      A = A2.T.dot(A2) for A positive definite, symmetric
  X : [n_dim] or [n_vec, n_dim] array
      vector(s) to be used for x.  If X has 2 dimensions, then each row will be
      added in turn.
  weight : float
      weight to be multiplied to each x x'. If negative, will use
      sign(weight) * sqrt(abs(weight)) instead of sqrt(weight).

  Returns
  -------
  A2 : [n_dim, n_dim array]
      cholesky decomposition of updated matrix

  Notes
  -----

  Code based on the following MATLAB snippet taken from Wikipedia on
  August 14, 2012::

      function [L] = cholupdate(L,x)
          p = length(x);
          x = x';
          for k=1:p
              r = sqrt(L(k,k)^2 + x(k)^2);
              c = r / L(k, k);
              s = x(k) / L(k, k);
              L(k, k) = r;
              L(k,k+1:p) = (L(k,k+1:p) + s*x(k+1:p)) / c;
              x(k+1:p) = c*x(k+1:p) - s*L(k, k+1:p);
          end
      end
  '''
  # make copies
  X = X.copy()
  A2 = A2.copy()

  # standardize input shape
  if len(X.shape) == 1:
      X = X[np.newaxis,:]
  n_vec, n_dim = X.shape

  # take sign of weight into account
  sign, weight = np.sign(weight), np.sqrt(np.abs(weight))
  X = weight * X

  for i in range(n_vec):
      x = X[i, :]
      for k in range(n_dim):
          r_squared = A2[k, k]**2 + sign * x[k]**2
          r = 0.0 if r_squared < 0 else np.sqrt(r_squared)
          c = r / A2[k, k]
          s = x[k] / A2[k, k]
          A2[k, k] = r
          A2[k, k+1:] = (A2[k, k+1:] + sign * s * x[k+1:]) / c
          x[k+1:] = c * x[k+1:] - s * A2[k, k+1:]
  return A2


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
        ##############################################
        # Same as this, but more stable (supposedly) #
        ##############################################
        # K = cross_sigma.dot(
        #     linalg.pinv(
        #         obs_sigma2_pred.T.dot(obs_sigma2_pred)
        #     )
        # )
        ##############################################

        # equivalent to this MATLAB code
        # K = (cross_sigma / obs_sigma2_pred.T) / obs_sigma2_pred
        K = linalg.lstsq(obs_sigma2_pred, cross_sigma.T)[0]
        K = linalg.lstsq(obs_sigma2_pred.T, K)[0]
        K = K.T

        # correct mu, sigma
        mu_filt = mu_pred + K.dot(z - obs_mu_pred)
        U = K.dot(obs_sigma2_pred)
        sigma2_filt = cholupdate(sigma2_pred, U.T, -1.0)
    else:
        # no corrections to be made
        mu_filt = mu_pred
        sigma2_filt = sigma2_pred
    return (mu_filt, sigma2_filt)


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

        #############################################
        # Same as this, but more stable (supposedly)#
        #############################################
        # smoother_gain = (
        #     sigma_pair.dot(linalg.pinv(sigma2_pred.T.dot(sigma2_pred)))
        # )
        #############################################
        smoother_gain = linalg.lstsq(sigma2_pred.T, sigma_pair.T)[0]
        smoother_gain = linalg.lstsq(sigma2_pred, smoother_gain)[0]
        smoother_gain = smoother_gain.T

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


class AdditiveUnscentedKalmanFilter(AUKF):
    r'''Implements the Unscented Kalman Filter with additive noise.
    Observations are assumed to be generated from the following process,

    .. math::

        x_0       &\sim \text{Normal}(\mu_0, \Sigma_0)  \\
        x_{t+1}   &=    f_t(x_t) + \text{Normal}(0, Q)  \\
        z_{t}     &=    g_t(x_t) + \text{Normal}(0, R)


    While less general the general-noise Unscented Kalman Filter, the Additive
    version is more computationally efficient with complexity :math:`O(Tn^3)`
    where :math:`T` is the number of time steps and :math:`n` is the size of
    the state space.

    Parameters
    ----------
    transition_functions : function or [n_timesteps-1] array of functions
        transition_functions[t] is a function of the state at time t and
        produces the state at time t+1. Also known as :math:`f_t`.
    observation_functions : function or [n_timesteps] array of functions
        observation_functions[t] is a function of the state at time t and
        produces the observation at time t. Also known as :math:`g_t`.
    transition_covariance : [n_dim_state, n_dim_state] array
        transition noise covariance matrix. Also known as :math:`Q`.
    observation_covariance : [n_dim_obs, n_dim_obs] array
        observation noise covariance matrix. Also known as :math:`R`.
    initial_state_mean : [n_dim_state] array
        mean of initial state distribution. Also known as :math:`\mu_0`.
    initial_state_covariance : [n_dim_state, n_dim_state] array
        covariance of initial state distribution. Also known as
        :math:`\Sigma_0`.
    n_dim_state: optional, integer
        the dimensionality of the state space. Only meaningful when you do not
        specify initial values for `transition_covariance`, or
        `initial_state_mean`, `initial_state_covariance`.
    n_dim_obs: optional, integer
        the dimensionality of the observation space. Only meaningful when you
        do not specify initial values for `observation_covariance`.
    random_state : optional, int or RandomState
        seed for random sample generation
    '''
    def filter(self, Z):
        '''Run Unscented Kalman Filter

        Parameters
        ----------
        Z : [n_timesteps, n_dim_state] array
            Z[t] = observation at time t.  If Z is a masked array and any of
            Z[t]'s elements are masked, the observation is assumed missing and
            ignored.

        Returns
        -------
        filtered_state_means : [n_timesteps, n_dim_state] array
            filtered_state_means[t] = mean of state distribution at time t
            given observations from times [0, t]
        filtered_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
            filtered_state_covariances[t] = covariance of state distribution at
            time t given observations from times [0, t]
        '''
        Z = self._parse_observations(Z)

        (transition_functions, observation_functions,
         transition_covariance, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        n_timesteps = Z.shape[0]

        # run square root filter
        (filtered_state_means, sigma2_filt) = (
            _additive_unscented_filter(
                initial_state_mean, initial_state_covariance,
                transition_functions, observation_functions,
                transition_covariance, observation_covariance,
                Z
            )
        )

        # reconstruct covariance matrices
        filtered_state_covariances = np.zeros(sigma2_filt.shape)
        for t in range(n_timesteps):
            filtered_state_covariances[t] = sigma2_filt[t].T.dot(sigma2_filt[t])

        return (filtered_state_means, filtered_state_covariances)

    def smooth(self, Z):
        '''Run Unscented Kalman Smoother

        Parameters
        ----------
        Z : [n_timesteps, n_dim_state] array
            Z[t] = observation at time t.  If Z is a masked array and any of
            Z[t]'s elements are masked, the observation is assumed missing and
            ignored.

        Returns
        -------
        smoothed_state_means : [n_timesteps, n_dim_state] array
            filtered_state_means[t] = mean of state distribution at time t
            given observations from times [0, n_timesteps-1]
        smoothed_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
            smoothed_state_covariances[t] = covariance of state distribution at
            time t given observations from times [0, n_timesteps-1]
        '''
        Z = self._parse_observations(Z)

        (transition_functions, observation_functions,
         transition_covariance, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        n_timesteps = Z.shape[0]

        # run filter, then smoother
        (filtered_state_means, sigma2_filt) = (
            _additive_unscented_filter(
                initial_state_mean, initial_state_covariance,
                transition_functions, observation_functions,
                transition_covariance, observation_covariance,
                Z
            )
        )
        (smoothed_state_means, sigma2_smooth) = (
            _additive_unscented_smoother(
                filtered_state_means, sigma2_filt,
                transition_functions, transition_covariance
            )
        )

        # reconstruction covariance matrices
        smoothed_state_covariances = np.zeros(sigma2_smooth.shape)
        for t in range(n_timesteps):
            smoothed_state_covariances[t] = (
                sigma2_smooth[t].T.dot(sigma2_smooth[t])
            )

        return (smoothed_state_means, smoothed_state_covariances)
