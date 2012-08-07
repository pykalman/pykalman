
=======================================================
`Kalman Filter <https://github.com/pykalman/pykalman>`_
=======================================================

Welcome to `kalman <https://github.com/pykalman/pykalman>`_, the dead-simple Kalman Filter, Kalman Smoother, and EM library for Python::

    >>> from kalman import KalmanFilter
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

    >>> I'll fill this in someday...


------------
Installation
------------

For a quick installation::

    $ easy_install kalman

:mod:`kalman` depends on the following modules,

* :mod:`numpy`     (for core functionality)
* :mod:`scipy`     (for core functionality)
* :mod:`Sphinx`    (for generating documentation)
* :mod:`numpydoc`  (for generating documentation)
* :mod:`nose`      (for running tests)

All of these and :mod:`kalman` can be installed using ``easy_install``::

    $ easy_install numpy scipy Sphinx numpydoc nose kalman

Alternatively, you can get the latest and greatest from `github
<https://github.com/pykalman/pykalman>`_::

    $ git clone git@github.com:pykalman/pykalman.git kalman
    $ cd kalman
    $ sudo python setup.py install


------------
User's Guide
------------

.. include:: users_guide.rst

---------------
Class Reference
---------------

.. include:: class_docs.rst
