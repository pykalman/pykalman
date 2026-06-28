"""Python implementations of Kalman Filters and Kalman Smoothers.

Inference methods for state-space estimation in continuous spaces.
"""

__version__ = "0.11.2"

from .standard import KalmanFilter
from .unscented import AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter
from .particle import ParticleFilter

__all__ = [
    "KalmanFilter",
    "AdditiveUnscentedKalmanFilter",
    "UnscentedKalmanFilter",
    "ParticleFilter",
    "datasets",
    "sqrt",
]
