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

from .standard import _last_dims, _determine_dimensionality


def _unscented_moments(points, weights_mu, weights_sigma, sigma_noise=None):
    '''Calculate the weighted mean and covariance of `points`

    Parameters
    ----------
    points : [2 * n_dim_state + 1, n_dim_state] array
        array where each row is a sigma point
    weights_mu : [2 * n_dim_state + 1] array
        weights used to calculate the mean
    weights_sigma : [2 * n_dim_state + 1] array
        weights used to calcualte the covariance
    sigma_noise : [n_dim_state, n_dim_state] array
        additive noise covariance matrix

    Returns
    -------
    mu : [n_dim_state] array
        approximate mean
    sigma : [n_dim_state, n_dim_state] array
        approximate covariance
    '''
    mu = points.T.dot(weights_mu)
    points_diff = points.T - mu[:, np.newaxis]
    sigma = points_diff.dot(np.diag(weights_sigma)).dot(points_diff.T)
    if sigma_noise is not None:
        sigma = sigma + sigma_noise
    return (mu.ravel(), sigma)


def _sigma_points(mu, sigma, alpha=1e-3, beta=2.0, kappa=0.0):
    '''Calculate "sigma points" used in Unscented Kalman Filter

    Parameters
    ----------
    mu : [n_dim] array
        Mean of multivariate normal distribution
    sigma : [n_dim, n_dim] array
        Covariance of multivariate normal
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

    # compute sqrt(sigma)
    sigma2 = linalg.cholesky(sigma).T

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
                         points_noise=None, sigma_noise=None):
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
    sigma_noise : [n_dim_2, n_dim_2] array
        covariance matrix for additive noise

    Returns
    =======
    points_pred : [n_points, n_dim_2] array
        points passed through f
    mu_pred : [n_dim_2] array
        empirical mean
    sigma_pred : [n_dim_2, n_dim_2] array
        empirical covariance
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
    (mu_pred, sigma_pred) = _unscented_moments(
        points_pred, weights_mean, weights_cov, sigma_noise)

    return (points_pred, mu_pred, sigma_pred)


def _unscented_correct(cross_sigma, mu_pred, sigma_pred, obs_mu_pred,
                       obs_sigma_pred, z):
    '''Correct predicted state estimates with an observation

    Parameters
    ----------
    cross_sigma : [n_dim_state, n_dim_obs] array
        cross-covariance between the state at time t given all observations
        from timesteps [0, t-1] and the observation at time t
    mu_pred : [n_dim_state] array
        mean of state at time t given observations from timesteps [0, t-1]
    sigma_pred : [n_dim_state, n_dim_state] array
        covariance of state at time t given observations from timesteps [0,
        t-1]
    obs_mu_pred : [n_dim_obs] array
        mean of observation at time t given observations from times [0, t-1]
    obs_sigma_pred : [n_dim_obs] array
        covariance of observation at time t given observations from times [0,
        t-1]
    z : [n_dim_obs] array
        observation at time t

    Returns
    -------
    mu_filt : [n_dim_state] array
        mean of state at time t given observations from time steps [0, t]
    sigma_filt : [n_dim_state, n_dim_state] array
        covariance of state at time t given observations from time steps [0, t]
    '''
    n_dim_state = len(mu_pred)
    n_dim_obs = len(obs_mu_pred)

    if not np.any(ma.getmask(z)):
        # calculate Kalman gain
        K = cross_sigma.dot(linalg.pinv(obs_sigma_pred))

        # correct mu, sigma
        mu_filt = mu_pred + K.dot(z - obs_mu_pred)
        sigma_filt = sigma_pred - K.dot(cross_sigma.T)
    else:
        # no corrections to be made
        mu_filt = mu_pred
        sigma_filt = sigma_pred
    return (mu_filt, sigma_filt)


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
        # Calculate sigma points for P(x_{t-1} | z_{0:t-1})
        if t == 0:
            mu, sigma = mu_0, sigma_0
        else:
            mu, sigma = mu_filt[t - 1], sigma_filt[t - 1]

        (points_state, weights_mu, weights_sigma) = _sigma_points(mu, sigma)

        # Calculate E[x_t | z_{0:t-1}], Var(x_t | z_{0:t-1})
        if t == 0:
            points_pred = points_state
            (mu_pred, sigma_pred) = (
                _unscented_moments(points_pred, weights_mu, weights_sigma)
            )
        else:
            f_t1 = _last_dims(f, t - 1, ndims=1)[0]
            (_, mu_pred, sigma_pred) = (
                _unscented_transform(f_t1, points_state,
                                     weights_mu, weights_sigma,
                                     sigma_noise=Q)
            )
            points_pred = _sigma_points(mu_pred, sigma_pred)[0]

        # Calculate E[z_t | z_{0:t-1}], Var(z_t | z_{0:t-1})
        g_t = _last_dims(g, t, ndims=1)[0]
        (obs_points_pred, obs_mu_pred, obs_sigma_pred) = (
            _unscented_transform(g_t, points_pred,
                                 weights_mu, weights_sigma,
                                 sigma_noise=R)
        )

        # Calculate Cov(x_t, z_t | z_{0:t-1})
        sigma_pair = (
            (points_pred - mu_pred).T
            .dot(np.diag(weights_sigma))
            .dot(obs_points_pred - obs_mu_pred)
        )

        # Calculate E[x_t | z_{0:t}], Var(x_t | z_{0:t})
        (mu_filt[t], sigma_filt[t]) = (
            _unscented_correct(sigma_pair, mu_pred, sigma_pred, obs_mu_pred,
                               obs_sigma_pred, Z[t])
        )

    return (mu_filt, sigma_filt)


def _additive_unscented_smoother(mu_filt, sigma_filt, f, Q):
    '''Apply the Unscented Kalman Filter assuming additiven noise

    Parameters
    ----------
    mu_filt : [T, n_dim_state] array
        mu_filt[t] = mean of state at time t given observations from times
        [0, t]
    sigma_filt : [T, n_dim_state, n_dim_state] array
        sigma_filt[t] = covariance of state at time t given observations from
        times [0, t]
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
        # get sigma points for state
        mu = mu_filt[t]
        sigma = sigma_filt[t]

        (points_state, weights_mu, weights_sigma) = (
            _sigma_points(mu, sigma)
        )

        # compute E[x_{t+1} | z_{0:t}], Var(x_{t+1} | z_{0:t})
        f_t = _last_dims(f, t, ndims=1)[0]
        (points_pred, mu_pred, sigma_pred) = (
            _unscented_transform(f_t, points_state, weights_mu,
                                 weights_sigma, sigma_noise=Q)
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


class UnscentedMixin(object):
    """Methods shared by all Unscented Kalman Filter implementations."""
    def __init__(self, transition_functions=None, observation_functions=None,
            transition_covariance=None, observation_covariance=None,
            initial_state_mean=None, initial_state_covariance=None,
            n_dim_state=None, n_dim_obs=None, random_state=None):

        self.transition_functions = transition_functions
        self.observation_functions = observation_functions
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs
        self.random_state = random_state

    def _initialize_parameters(self):
        """Retrieve parameters if they exist, else replace with defaults"""

        # determine size of state and observation space
        n_dim_state = _determine_dimensionality(
            [(self.transition_covariance, array2d, -2),
             (self.initial_state_covariance, array2d, -2),
             (self.initial_state_mean, array1d, -1)],
            self.n_dim_state
        )
        n_dim_obs = _determine_dimensionality(
            [(self.observation_covariance, array2d, -2)],
            self.n_dim_obs
        )

        # initialize parameters
        transition_functions = (
            array1d(lambda state, noise: state + noise)
            if self.transition_functions is None
            else array1d(self.transition_functions)
        )
        transition_covariance = (
            np.eye(n_dim_state)
            if self.transition_covariance is None
            else array2d(self.transition_covariance)
        )
        observation_functions = (
            array1d(lambda state, noise: state + noise)
            if self.observation_functions is None
            else array1d(self.observation_functions)
        )
        observation_covariance = (
            np.eye(n_dim_obs)
            if self.observation_covariance is None
            else array2d(self.observation_covariance)
        )
        initial_state_mean = (
            np.zeros(n_dim_state)
            if self.initial_state_mean is None
            else array1d(self.initial_state_mean)
        )
        initial_state_covariance = (
            np.eye(n_dim_state)
            if self.initial_state_covariance is None
            else array2d(self.initial_state_covariance)
        )
        return (transition_functions, observation_functions,
            transition_covariance, observation_covariance,
            initial_state_mean, initial_state_covariance)

    def _parse_observations(self, obs):
        """Safely convert observations to their expected format"""
        obs = ma.atleast_2d(obs)
        if obs.shape[0] == 1 and obs.shape[1] > 1:
            obs = obs.T
        return obs


class UnscentedKalmanFilter(UnscentedMixin):
    r'''Implements the General (aka Augmented) Unscented Kalman Filter governed
    by the following equations,

    .. math::

        x_0       &\sim \text{Normal}(\mu_0, \Sigma_0)  \\
        x_{t+1}   &=    f_t(x_t, \text{Normal}(0, Q))   \\
        z_{t}     &=    g_t(x_t, \text{Normal}(0, R))

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
    transition_functions : function or [n_timesteps-1] array of functions
        transition_functions[t] is a function of the state and the transition
        noise at time t and produces the state at time t+1.  Also known as
        :math:`f_t`.
    observation_functions : function or [n_timesteps] array of functions
        observation_functions[t] is a function of the state and the observation
        noise at time t and produces the observation at time t.  Also known as
        :math:`g_t`.
    transition_covariance : [n_dim_state, n_dim_state] array
        transition noise covariance matrix. Also known as :math:`Q`.
    observation_covariance : [n_dim_obs, n_dim_obs] array
        observation noise covariance matrix. Also known as :math:`R`.
    initial_state_mean : [n_dim_state] array
        mean of initial state distribution. Also known as :math:`\mu_0`
    initial_state_covariance : [n_dim_state, n_dim_state] array
        covariance of initial state distribution. Also known as
        :math:`\Sigma_0`
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
    def sample(self, n_timesteps, initial_state=None, random_state=None):
        '''Sample from model defined by the Unscented Kalman Filter

        Parameters
        ----------
        n_timesteps : int
            number of time steps
        initial_state : optional, [n_dim_state] array
            initial state.  If unspecified, will be sampled from initial state
            distribution.
        random_state : optional, int or Random
            random number generator
        '''
        (transition_functions, observation_functions,
         transition_covariance, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        n_dim_state = transition_covariance.shape[-1]
        n_dim_obs = observation_covariance.shape[-1]

        # logic for instantiating rng
        if random_state is None:
            rng = check_random_state(self.random_state)
        else:
            rng = check_random_state(random_state)

        # logic for selecting initial state
        if initial_state is None:
            initial_state = rng.multivariate_normal(
                initial_state_mean, initial_state_covariance
            )

        # logic for generating samples
        x = np.zeros((n_timesteps, n_dim_state))
        z = np.zeros((n_timesteps, n_dim_obs))
        for t in range(n_timesteps):
            if t == 0:
                x[0] = initial_state
            else:
                transition_function = (
                    _last_dims(transition_functions, t - 1, ndims=1)[0]
                )
                transition_noise = (
                    rng.multivariate_normal(
                        np.zeros(n_dim_state),
                        transition_covariance.newbyteorder('=')
                    )
                )
                x[t] = transition_function(x[t - 1], transition_noise)

            observation_function = (
                _last_dims(observation_functions, t, ndims=1)[0]
            )
            observation_noise = (
                rng.multivariate_normal(
                    np.zeros(n_dim_obs),
                    observation_covariance.newbyteorder('=')
                )
            )
            z[t] = observation_function(x[t], observation_noise)

        return (x, ma.asarray(z))

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
            filtered_state_means[t] = mean of state distribution at time t given
            observations from times [0, t]
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

        (filtered_state_means, filtered_state_covariances) = (
            _augmented_unscented_filter(
                initial_state_mean, initial_state_covariance,
                transition_functions, observation_functions,
                transition_covariance, observation_covariance,
                Z
            )
        )

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
            filtered_state_means[t] = mean of state distribution at time t given
            observations from times [0, n_timesteps-1]
        smoothed_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
            filtered_state_covariances[t] = covariance of state distribution at
            time t given observations from times [0, n_timesteps-1]
        '''
        Z = self._parse_observations(Z)

        (transition_functions, observation_functions,
         transition_covariance, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        (filtered_state_means, filtered_state_covariances) = self.filter(Z)
        (smoothed_state_means, smoothed_state_covariances) = (
            _augmented_unscented_smoother(
                filtered_state_means, filtered_state_covariances,
                transition_functions, transition_covariance
            )
        )

        return (smoothed_state_means, smoothed_state_covariances)


class AdditiveUnscentedKalmanFilter(UnscentedMixin):
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
    def sample(self, n_timesteps, initial_state=None, random_state=None):
        '''Sample from model defined by the Unscented Kalman Filter

        Parameters
        ----------
        n_timesteps : int
            number of time steps
        initial_state : optional, [n_dim_state] array
            initial state.  If unspecified, will be sampled from initial state
            distribution.
        '''
        (transition_functions, observation_functions,
         transition_covariance, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        n_dim_state = transition_covariance.shape[-1]
        n_dim_obs = observation_covariance.shape[-1]

        # logic for instantiating rng
        if random_state is None:
            rng = check_random_state(self.random_state)
        else:
            rng = check_random_state(random_state)

        # logic for selecting initial state
        if initial_state is None:
            initial_state = (
                rng.multivariate_normal(
                    initial_state_mean,
                    initial_state_covariance
                )
            )

        # logic for generating samples
        x = np.zeros((n_timesteps, n_dim_state))
        z = np.zeros((n_timesteps, n_dim_obs))
        for t in range(n_timesteps):
            if t == 0:
                x[0] = initial_state
            else:
                transition_function = (
                    _last_dims(transition_functions, t - 1, ndims=1)[0]
                )
                transition_noise = (
                    rng.multivariate_normal(
                        np.zeros(n_dim_state),
                        transition_covariance.newbyteorder('=')
                    )
                )
                x[t] = transition_function(x[t - 1]) + transition_noise

            observation_function = (
                _last_dims(observation_functions, t, ndims=1)[0]
            )
            observation_noise = (
                rng.multivariate_normal(
                    np.zeros(n_dim_obs),
                    observation_covariance.newbyteorder('=')
                )
            )
            z[t] = observation_function(x[t]) + observation_noise

        return (x, ma.asarray(z))

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
            filtered_state_means[t] = mean of state distribution at time t given
            observations from times [0, t]
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

        (filtered_state_means, filtered_state_covariances) = (
            _additive_unscented_filter(
                initial_state_mean, initial_state_covariance,
                transition_functions, observation_functions,
                transition_covariance, observation_covariance,
                Z
            )
        )

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
            filtered_state_means[t] = mean of state distribution at time t given
            observations from times [0, n_timesteps-1]
        smoothed_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
            filtered_state_covariances[t] = covariance of state distribution at
            time t given observations from times [0, n_timesteps-1]
        '''
        Z = ma.asarray(Z)

        (transition_functions, observation_functions,
         transition_covariance, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        (filtered_state_means, filtered_state_covariances) = self.filter(Z)
        (smoothed_state_means, smoothed_state_covariances) = (
            _additive_unscented_smoother(
                filtered_state_means, filtered_state_covariances,
                transition_functions, transition_covariance
            )
        )

        return (smoothed_state_means, smoothed_state_covariances)
