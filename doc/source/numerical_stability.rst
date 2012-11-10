.. currentmodule:: pykalman.sqrt

The Kalman Filter and Unscented Kalman Filter are occasionally prone to failure
due to numerical errors, causing the algorithm to return incorrect estimates or
fail entirely. These errors typically surface when one or more of the
:attr:`filtered_state_covariances` have very, very small eigenvalues. In order
to remain a valid covariance matrix, all of the eigenvalues of
:attr:`filtered_state_covariances` must remain positive -- in other words, must
remain `positive definite
<http://en.wikipedia.org/wiki/Positive-definite_matrix>`_. Unfortunately, the
Kalman Filter update equations involve subtracting two positive definite
matrices which, due to numerical error, can result in a negative definite
matrix that is no longer a proper covariance matrix!  Once that happens, the
filter can no longer continue.

To combat this, two versions of the Kalman Filter and a version of the
Additive-noise Unscented Kalman Filter are implemented which use factorized
versions of :attr:`filtered_state_covariances`. Unlike their standard
counterparts, these implementations are far less susceptible to numerical
error, but do require roughly 20% more time to run.

Cholesky-Based Kalman Filter
============================

The first is :class:`CholeskyKalmanFilter`, which uses the Cholesky
factorization to decompose a covariance matrix into the product of two lower
triangular matrices, that is

.. math::

    \Sigma = L L^{T}

Since :math:`L` is used instead of :math:`\Sigma`, you can't accidentally
create a matrix that's negative definite. :class:`CholeskyKalmanFilter` is
designed to be a drop-in replacement for :class:`KalmanFilter`::

    >>> import numpy as np
    >>> from pykalman import KalmanFilter as KF1              # standard Kalman Filter formulation
    >>> from pykalman.sqrt import CholeskyKalmanFilter as KF2 # LL' decomposition Kalman Filter
    >>> from numpy.testing import assert_array_almost_equal
    >>> transition_matrix = [[1, 1], [0, 1]]                  # parameters
    >>> observation_matrix = [[0.1, 0.5], [-0.3, 0.0]]
    >>> measurements = [[1,0], [0,0], [0,1]]                  # measurements
    >>> kf1 = KF1(transition_matrices=transition_matrix, observation_matrices=observation_matrix)
    >>> kf2 = KF2(transition_matrices=transition_matrix, observation_matrices=observation_matrix)
    >>> assert_array_almost_equal(kf1.filter(measurements)[0], kf2.filter(measurements)[0])

Currently only :func:`CholeskyKalmanFilter.filter` makes use the Cholesky
factorization, so the smoother may still suffer numerical instability.

.. topic:: References:

 * Salzmann, M. A. Some Aspects of Kalman Filtering. August 1988. Page 31.


UDU'-Based Kalman Filter
========================

A second implementation named after its inventor, G. J. Bierman, is the
:class:`BiermanKalmanFilter`. This version is based on a less common
matrix decomposition,

.. math::

    \Sigma = U D U^{T}

Here :math:`U` is an upper triangular matrix with 1s along the diagonal and
:math:`D` is diagonal matrix. The beauty of this representation is that the
Kalman Filter update doesn't require reconstructing :math:`\Sigma`. To use the
:class:`BiermanKalmanFilter`, one only need import it instead of the
:class:`CholeskyKalmanFilter` in the previous example::

    >>> from pykalman.sqrt import BiermanKalmanFilter as KF2

Currently only :func:`BiermanKalmanFilter.filter` makes use the :math:`UDU^{T}`
factorization, so the smoother may still suffer numerical instability.

.. topic:: References:

 * Gibbs, Bruce P. Advanced Kalman Filtering, Least-Squares, and Modeling: A
   Practical Handbook. Page 396

Square Root Unscented Kalman Filter
===================================

In 2001, the original inventors of the Unscented Kalman Filter derived a
"square root" form based on the Cholesky Factorization. Like its standard
Kalman Filter counterpart, the "square root" form is less likely to suffer from
numerical errors.  Its use is identical to the typical
:class:`AdditiveUnscentedKalmanFilter`::

    >>> from pykalman.sqrt import AdditiveUnscentedKalmanFilter

The implementations of both :func:`AdditiveUnscentedKalmanFilter.filter` and
:func:`AdditiveUnscentedKalmanFilter.smooth` make use of the Cholesky
factorization.

.. topic:: References:

 * Terejanu, G.A. Towards a Decision-Centric Framework for Uncertainty
   Propagation and Data Assimilation. 2010.
 * Van Der Merwe, R. and Wan, E.A. The Square-Root Unscented Kalman Filter for
   State and Parameter-Estimation. 2001.
