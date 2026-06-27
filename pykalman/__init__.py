"""Python implementations of Kalman Filters and Kalman Smoothers.

Inference methods for state-space estimation in continuous spaces.
"""

__version__ = "0.11.2"

from .extended import ExtendedKalmanFilter
from .standard import KalmanFilter
from .unscented import AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter

__all__ = [
    "ExtendedKalmanFilter",
    "KalmanFilter",
    "AdditiveUnscentedKalmanFilter",
    "UnscentedKalmanFilter",
    "datasets",
    "sqrt",
]
