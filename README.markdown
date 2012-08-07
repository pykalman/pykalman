Kalman Filters for Python
=========================

Welcome to `kalman`, the dead-simple Kalman Filter, Kalman Smoother, and EM library for Python:

    >>> from kalman import KalmanFilter
    >>> import numpy as np
    >>> kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
    >>> measurements = np.asarray([[1,0], [0,0], [0,1]])  # 3 observations
    >>> kf = kf.em(measurements, n_iter=5)
    >>> (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    >>> (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

Also included is support for missing measurements:

    >>> from numpy import ma
    >>> measurements = ma.asarray(measurements)
    >>> measurements[1] = ma.masked   # measurement at timestep 1 is unobserved
    >>> kf = kf.em(measurements, n_iter=5)
    >>> (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    >>> (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

And for the non-linear dynamics via the `UnscentedKalmanFilter`


Installation
============

`kalman` depends on the following libraries

* numpy     (for core functionality)
* scipy     (for core functionality)
* Sphinx    (for generating documentation)
* numpydoc  (for generating documentation)
* nose      (for running tests)

All of these and `kalman` can be installed via `easy_install`:

    $ easy_install numpy scipy Sphinx numpydoc nose kalman


Examples
========

Examples of all of `kalman`'s functionality can be found in the scripts in the examples/ folder.
