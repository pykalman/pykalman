"""Python implementations of Kalman Filters and Kalman Smoothers.

Inference methods for state-space estimation in continuous spaces.
"""

__version__ = "0.10.1"

from .standard import KalmanFilter
from .unscented import AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter

__all__ = [
    "KalmanFilter",
    "AdditiveUnscentedKalmanFilter",
    "UnscentedKalmanFilter",
    "datasets",
    "sqrt",
]
