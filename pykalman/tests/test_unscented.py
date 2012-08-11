import numpy as np
from numpy import ma
from numpy.testing import assert_array_almost_equal

from nose.tools import assert_true

from pykalman import AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter
from pykalman.datasets import load_robot

data = load_robot()


def build_unscented_filter(cls):
    '''Instantiate the Unscented Kalman Filter'''
    # build transition functions
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[0.5, -0.3]])
    if cls == UnscentedKalmanFilter:
        f = lambda x, y: A.dot(x) + y
        g = lambda x, y: C.dot(x) + y
    elif cls == AdditiveUnscentedKalmanFilter:
        f = lambda x: A.dot(x)
        g = lambda x: C.dot(x)

    x = np.array([1, 1])
    P = np.array([[1, 0.1], [0.1, 1]])

    Q = np.eye(2) * 2
    R = 0.5

    # build filter
    kf = cls(f, g, Q, R, x, P, random_state=0)

    return kf


def check_unscented_prediction(method, mu_true, sigma_true):
    '''Check output of a method against true mean and covariances'''
    Z = ma.array([0, 1, 2, 3], mask=[True, False, False, False])
    (mu_est, sigma_est) = method(Z)
    mu_est, sigma_est = mu_est[1:], sigma_est[1:]

    assert_array_almost_equal(mu_true, mu_est, decimal=8)
    assert_array_almost_equal(sigma_true, sigma_est, decimal=8)


def test_unscented_sample():
    kf = build_unscented_filter(UnscentedKalmanFilter)
    (x, z) = kf.sample(100)

    assert_true(x.shape == (100, 2))
    assert_true(z.shape == (100, 1))


def test_unscented_filter():
    # true unscented mean, covariance, as calculated by a MATLAB ukf_predict3
    # and ukf_update3 available from
    # http://becs.aalto.fi/en/research/bayes/ekfukf/
    mu_true = np.zeros((3, 2), dtype=float)
    mu_true[0] = [2.35637583900053, 0.92953020131845]
    mu_true[1] = [4.39153258583784, 1.15148930114305]
    mu_true[2] = [6.71906243764755, 1.52810614201467]

    sigma_true = np.zeros((3, 2, 2), dtype=float)
    sigma_true[0] = [[2.09738255033564, 1.51577181208054],
                     [1.51577181208054, 2.91778523489934]]
    sigma_true[1] = [[3.62532578216913, 3.14443733560803],
                     [3.14443733560803, 4.65898912348045]]
    sigma_true[2] = [[4.3902465859811, 3.90194406652627],
                     [3.90194406652627, 5.40957304471697]]

    check_unscented_prediction(
        build_unscented_filter(UnscentedKalmanFilter).filter,
        mu_true, sigma_true
    )


def test_unscented_smoother():
    # true unscented mean, covariance, as calculated by a MATLAB urts_smooth2
    # available in http://becs.aalto.fi/en/research/bayes/ekfukf/
    mu_true = np.zeros((3, 2), dtype=float)
    mu_true[0] = [2.92725011530645, 1.63582509442842]
    mu_true[1] = [4.87447429684622,  1.6467868915685]
    mu_true[2] = [6.71906243764755, 1.52810614201467]

    sigma_true = np.zeros((3, 2, 2), dtype=float)
    sigma_true[0] = [[0.993799756492982, 0.216014513083516],
                     [0.216014513083516, 1.25274857496387]]
    sigma_true[1] = [[1.57086880378025, 1.03741785934464],
                     [1.03741785934464, 2.49806235789068]]
    sigma_true[2] = [[4.3902465859811, 3.90194406652627],
                     [3.90194406652627, 5.40957304471697]]

    check_unscented_prediction(
        build_unscented_filter(UnscentedKalmanFilter).smooth,
        mu_true, sigma_true
    )


def test_additive_sample():
    kf = build_unscented_filter(AdditiveUnscentedKalmanFilter)
    (x, z) = kf.sample(100)

    assert_true(x.shape == (100, 2))
    assert_true(z.shape == (100, 1))


def test_additive_filter():
    # true unscented mean, covariance, as calculated by a MATLAB ukf_predict1
    # and ukf_update1 available from
    # http://becs.aalto.fi/en/research/bayes/ekfukf/
    mu_true = np.zeros((3, 2), dtype=float)
    mu_true[0] = [2.3563758389014, 0.929530201358681]
    mu_true[1] = [4.39153258609087, 1.15148930112108]
    mu_true[2] = [6.71906243585852, 1.52810614139809]

    sigma_true = np.zeros((3, 2, 2), dtype=float)
    sigma_true[0] = [[2.09738255033572, 1.51577181208044],
                     [1.51577181208044, 2.91778523489926]]
    sigma_true[1] = [[3.62532578216869, 3.14443733560774],
                     [3.14443733560774, 4.65898912348032]]
    sigma_true[2] = [[4.39024658597909, 3.90194406652556],
                     [3.90194406652556, 5.40957304471631]]

    check_unscented_prediction(
        build_unscented_filter(AdditiveUnscentedKalmanFilter).filter,
        mu_true, sigma_true
    )


def test_additive_smoother():
    # true unscented mean, covariance, as calculated by a MATLAB urts_smooth1
    # available in http://becs.aalto.fi/en/research/bayes/ekfukf/
    mu_true = np.zeros((3, 2), dtype=float)
    mu_true[0] = [2.92725011499923, 1.63582509399207]
    mu_true[1] = [4.87447429622188, 1.64678689063005]
    mu_true[2] = [6.71906243585852, 1.52810614139809]

    sigma_true = np.zeros((3, 2, 2), dtype=float)
    sigma_true[0] = [[0.99379975649288, 0.21601451308325],
                     [0.21601451308325, 1.25274857496361]]
    sigma_true[1] = [[1.570868803779,   1.03741785934372],
                     [1.03741785934372, 2.49806235789009]]
    sigma_true[2] = [[4.39024658597909, 3.90194406652556],
                     [3.90194406652556, 5.40957304471631]]

    check_unscented_prediction(
        build_unscented_filter(UnscentedKalmanFilter).smooth,
        mu_true, sigma_true
    )
