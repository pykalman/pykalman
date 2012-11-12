
=======================================================
`pykalman <https://github.com/pykalman/pykalman>`_
=======================================================

Welcome to `pykalman <https://github.com/pykalman/pykalman>`_, the dead-simple Kalman Filter, Kalman Smoother, and EM library for Python::

    >>> from pykalman import KalmanFilter
    >>> import numpy as np
    >>> kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
    >>> measurements = np.asarray([[1,0], [0,0], [0,1]])  # 3 observations
    >>> kf = kf.em(measurements, n_iter=5)
    >>> (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    >>> (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

Also included is support for missing measurements::

    >>> from numpy import ma
    >>> measurements = ma.asarray(measurements)
    >>> measurements[1] = ma.masked   # measurement at timestep 1 is unobserved
    >>> kf = kf.em(measurements, n_iter=5)
    >>> (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    >>> (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

And for the non-linear dynamics via the :class:`UnscentedKalmanFilter`::

    >>> from pykalman import UnscentedKalmanFilter
    >>> ukf = UnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, observation_covariance=0.1)
    >>> (filtered_state_means, filtered_state_covariances) = ukf.filter([0, 1, 2])
    >>> (smoothed_state_means, smoothed_state_covariances) = ukf.smooth([0, 1, 2])

And for online state estimation::

    >>> for t in range(1, 3):
    ...     filtered_state_means[t], filtered_state_covariances[t] = \
    ...         kf.filter_update(filtered_state_means[t-1], filtered_state_covariances[t-1], measurements[t])

And for numerically robust "square root" filters::

    >>> from pykalman.sqrt import CholeskyKalmanFilter, AdditiveUnscentedKalmanFilter
    >>> kf = CholeskyKalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
    >>> ukf = AdditiveUnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, observation_covariance=0.1)

------------
Installation
------------

For a quick installation::

    $ easy_install pykalman

:mod:`pykalman` depends on the following modules,

* :mod:`numpy`     (for core functionality)
* :mod:`scipy`     (for core functionality)
* :mod:`Sphinx`    (for generating documentation)
* :mod:`numpydoc`  (for generating documentation)
* :mod:`nose`      (for running tests)

All of these and :mod:`pykalman` can be installed using ``easy_install``::

    $ easy_install numpy scipy Sphinx numpydoc nose pykalman

Alternatively, you can get the latest and greatest from `github
<https://github.com/pykalman/pykalman>`_::

    $ git clone git@github.com:pykalman/pykalman.git pykalman
    $ cd pykalman
    $ sudo python setup.py install


--------------------------
Kalman Filter User's Guide
--------------------------

.. include:: kf_users_guide.rst

------------------------------------
Unscented Kalman Filter User's Guide
------------------------------------

.. include:: ukf_users_guide.rst

-----------------------------
Dealing with Numerical Errors
-----------------------------

.. include:: numerical_stability.rst

---------------
Class Reference
---------------

.. include:: class_docs.rst
