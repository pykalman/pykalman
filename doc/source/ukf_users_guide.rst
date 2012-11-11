.. currentmodule:: pykalman

Like the Kalman Filter, the Unscented Kalman Filter is an unsupervised
algorithm for tracking a single target in a continuous state space. The
difference is that while the Kalman Filter restricts dynamics to affine
functions, the Unscented Kalman Filter is designed to operate under arbitrary
dynamics.

The advantages of the Unscented Kalman Filter implemented here are:

    + Ability to handle non-affine state transition and observation functions

    + Ability to handle not-quite-Gaussian noise models

    + Same computational complexity as the standard Kalman Filter

The disadvantages are:

    + No method for learning parameters

    + Lack of theoretical guarantees on performance

    + Inability to handle extremely non-Gaussian noise


Basic Usage
===========

Like :class:`KalmanFilter`, two methods are provided in
:class:`UnscentedKalmanFilter` for tracking targets:
:func:`UnscentedKalmanFilter.filter` and :func:`UnscentedKalmanFilter.smooth`.
At this point no algorithms have been implemented for inferring parameters, so
they must be specified by hand at instantiation.

In order to apply these algorithms, one must specify a subset of the following,

    +---------------------------------------+--------------------------------+---------------------+
    | Variable Name                         | Mathematical Notation          | Default             |
    +---------------------------------------+--------------------------------+---------------------+
    | :attr:`transition_functions[t]`       | :math:`f_t`                    | state plus noise    |
    +---------------------------------------+--------------------------------+---------------------+
    | :attr:`observation_functions[t]`      | :math:`g_t`                    | state plus noise    |
    +---------------------------------------+--------------------------------+---------------------+
    | :attr:`transition_covariance`         | :math:`Q`                      | identity            |
    +---------------------------------------+--------------------------------+---------------------+
    | :attr:`observation_covariance`        | :math:`R`                      | identity            |
    +---------------------------------------+--------------------------------+---------------------+
    | :attr:`initial_state_mean`            | :math:`\mu_0`                  | zero                |
    +---------------------------------------+--------------------------------+---------------------+
    | :attr:`initial_state_covariance`      | :math:`\Sigma_0`               | identity            |
    +---------------------------------------+--------------------------------+---------------------+

If parameters are left unspecified, they will be replaced by their defaults.
One also has the option of simply specifying :attr:`n_dim_state` or
:attr:`n_dim_obs` if the size of the state or observation space cannot be
inferred directly.

The state transition function and observation function have replaced the
transition matrix/offset and observation matrix/offset from the original
:class:`KalmanFilter`, respectively. Both must take in the current state and
some Gaussian-sampled noise and return the next state/current observation. For
example, if noise were multiplicative instead of additive, the following would
be valid::

    >>> def f(current_state, transition_noise):
    ...     return current_state * transition_noise
    ...
    >>> def g(current_state, observation_noise):
    ...     return current_state * observation_noise

Once defined, the :class:`UnscentedKalmanFilter` can be used to extract
estimated state and covariance matrices over the hidden state::

    >>> from pykalman import UnscentedKalmanFilter
    >>> import numpy as np
    >>> def f(state, noise):
    ...     return state + np.sin(noise)
    ...
    >>> def g(state, noise):
    ...     return state + np.cos(noise)
    ...
    >>> ukf = UnscentedKalmanFilter(f, g, observation_covariance=0.1)
    >>> ukf.smooth([0, 1, 2])[0]
    array([[-0.94034641],
           [ 0.05002316],
           [ 1.04502498]])

Methods for online estimation are also included::

    >>> means, covariances = ukf.filter([0,1])
    >>> next_mean, next_covariance = ukf.filter_update(means[-1], covariances[-1], [2])

If the :class:`UnscentedKalmanFilter` is instantiated with an array of
functions for :attr:`transition_functions` or :attr:`observation_functions`,
then the function is assumed to vary with time. Currently there is no support
for time-varying covariance matrices.


Which Unscented Kalman Filter is for Me?
----------------------------------------

Though only :class:`UnscentedKalmanFilter` was mentioned in the previous
section, there exists another class specifically designed for the case when
noise is additive, :class:`AdditiveUnscentedKalmanFilter`. While more
restrictive, this class offers reduced computational complexity
(:math:`O(Tn^3)` vs. :math:`O(T(2n+m)^3)` for state space with dimensionality
:math:`n`, observation space with dimensionality :math:`m`) and better
numerical stability. When at all possible, the
:class:`AdditiveUnscentedKalmanFilter` should be preferred to its counterpart.

To reflect the restriction on how noise is integrated, the
:class:`AdditiveUnscentedKalmanFilter` uses state transition and observation
functions with slightly different arguments::

    def f(current_state):
        ...

    def g(current_state):
        ...

Notice that the transition/observation noise is no longer an argument. Its
effect will be taken care of at later points in the algorithm without any need
for your explicit input.

Finally, users should note that the :class:`UnscentedKalmanFilter` can
potentially suffer from collapse of the covariance matrix to zero.
Algorithmically, this means that the UnscentedKalmanFilter is one hundred
percent sure of the state and that no noise is left in the system. In order to
avoid this, one must ensure that even for small amounts of noise,
:attr:`transition_functions` and :attr:`observation_functions` output different
values for the same current state.


Choosing Parameters
-------------------

The majority of advice on choosing parameters in Kalman Filter section apply to
the Unscented Kalman Filter except that there is no method for learning
parameters and the following code snippet defines the probabilistic model the
Unscented Kalman Filter (approximately) solves,

.. code-block:: python

    from scipy.stats import norm
    import numpy as np
    states = np.zeros((n_timesteps, n_dim_state))
    measurements = np.zeros((n_timesteps, n_dim_obs))
    for t in range(n_timesteps-1):
       if t == 0:
          states[t] = norm.rvs(initial_state_mean, np.sqrt(initial_state_covariance))
          measurements[t] = (
              observation_function(
                  states[t],
                  norm.rvs(0, np.sqrt(observation_covariance))
              )
          )
      states[t+1] = (
          transition_function(
              states[t],
              norm.rvs(0, np.sqrt(transition_covariance))
          )
      )
      measurements[t+1] = (
          observation_function(
              states[t+1],
              norm.rvs(0, np.sqrt(observation_covariance))
          )
      )

Missing Measurements
--------------------

The :class:`UnscentedKalmanFilter` and :class:`AdditiveUnscentedKalmanFilter`
have the same support for missing measurements that the original
:class:`KalmanFilter` class supports. Usage is precisely the same.


Mathematical Formulation
========================

The mathematical formulation of the model the Unscented Kalman Filter is based
on is unsurprisingly similar to that of the standard Kalman Filter. In this
section, we will again adopt the notation :math:`x_t` to describe the (unknown)
state at time step :math:`t` and :math:`z_t` to denote the (known) observation.
The parameters of the Unscented Kalman Filter are as follows,

    +-----------------------------+------------------+
    | Parameter Name              | Notation         |
    +-----------------------------+------------------+
    | `transition_functions`      | :math:`f`        |
    +-----------------------------+------------------+
    | `observation_functions`     | :math:`g`        |
    +-----------------------------+------------------+
    | `transition_covariance`     | :math:`Q`        |
    +-----------------------------+------------------+
    | `observation_covariance`    | :math:`R`        |
    +-----------------------------+------------------+
    | `initial_state_mean`        | :math:`\mu_0`    |
    +-----------------------------+------------------+
    | `initial_state_covariance`  | :math:`\Sigma_0` |
    +-----------------------------+------------------+


To estimate :math:`x_t` using the above and :math:`z_t`, we must describe how
these parameters effect :math:`x_t` and :math:`z_t`.  The two models used, as
implemented by :class:`AdditiveUnscentedKalmanFilter` and
:class:`UnscentedKalmanFilter` respectively are as follows,

In the case of the additive noise model,

.. math::

    x_0               & \sim \text{Gaussian}(\mu_0, \Sigma_0)   \\
    x_{t+1}           & = f_t(x_t) + \epsilon_{t+1}^{1}         \\
    z_{t}             & = g_t(x_t) + \epsilon_{t}^2             \\
    \epsilon_t^1      & \sim \text{Gaussian}(0, Q)              \\
    \epsilon_{t}^2    & \sim \text{Gaussian}(0, R)

The general model is nearly the same, except with the following modifications,

.. math::

    x_{t+1}           & = f_t(x_t, \epsilon_{t+1}^1)            \\
    z_{t}             & = g_t(x_t, \epsilon_{t}^2)

In the standard Kalman Filter case, we could be certain that the probability
distribution over :math:`x_t` is always a Gaussian distribution, even after
observing :math:`z_t`. In the above scenarios that is no longer the case, but
we make the *approximation* that it is. If :math:`f_t` and :math:`g_t` are
*almost* linear, then the Unscented Kalman Filter will be a good approximation.
