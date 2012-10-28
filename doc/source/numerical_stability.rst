.. currentmodule:: pykalman

The Kalman Filter and Unscented Kalman Filter are occasionally prone to failure
due to numerical errors, causing the algorithm to return incorrect estimates or
fail entirely. These errors typically surface when the covariance matrices have
very, very small entries or when the observations strongly disagree with what
the model predicts (i.e. your model is wrong!).

To combat this, the "Square Root" Kalman Filter and Unscented Kalman Filter
have been implemented in :mod:`pykalman.sqrt`. Unlike the traditional Kalman
Filter / Unscented Kalman Filter which propagate the mean and covariance of a
multivariate Normal distribution, the "Square Root" versions propagate their
Cholesky factorization directly. What this means is that we can never end up
with a matrix that has negative eigenvalues (this is what ultimately causes the
algorithm to fail).

Currently, the :class:`AdditiveUnscentedKalmanFilter`'s :func:`smooth` and
:func:`filter` methods and the :class:`KalmanFilter`'s :func:`filter` have been
implemented in "Square Root" form. Their interfaces are identical to that of
their respective "normal" implementations::

    >>> from pykalman import KalmanFilter as KF
    >>> from pykalman.sqrt import KalmanFilter as KF_SQRT
    >>> from numpy.testing import assert_array_almost_equal
    >>> kf1 = KF(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
    >>> kf2 = KF_SQRT(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
    >>> measurements = np.asarray([[1,0], [0,0], [0,1]])  # 3 observations
    >>> assert_array_almost_equal(kf1.filter(measurements)[0], kf2.filter(measurements)[0])
    >>> assert_array_almost_equal(kf1.smooth(measurements)[0], kf2.smooth(measurements)[0])
