# pykalman-bardo (reborn pykalman)

**Notice**: This a fork of original [pykalman](https://github.com/pykalman/pykalman) package.
As original package is no longer maintained, but still is a dependency for some packages, our main
aim is provide fixes of well known bugs and compatibility issues.

Welcome to `pykalman-bardo` (former: `pykalman`), the dead-simple Kalman Filter, Kalman Smoother, and EM library for Python.

## Installation

For a quick installation::

```bash
pip install pykalman-bardo
```

Alternatively, you can setup from source:

```bash
pip install .
```

## Usage

``` python
from pykalman import KalmanFilter
import numpy as np
kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
measurements = np.asarray([[1,0], [0,0], [0,1]])  # 3 observations
kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
```

Also included is support for missing measurements:

```python
from numpy import ma
measurements = ma.asarray(measurements)
measurements[1] = ma.masked   # measurement at timestep 1 is unobserved
kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
```

And for the non-linear dynamics via the `UnscentedKalmanFilter`:

```python
from pykalman import UnscentedKalmanFilter
ukf = UnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, transition_covariance=0.1)
(filtered_state_means, filtered_state_covariances) = ukf.filter([0, 1, 2])
(smoothed_state_means, smoothed_state_covariances) = ukf.smooth([0, 1, 2])
```

And for online state estimation:

```python
 for t in range(1, 3):
    filtered_state_means[t], filtered_state_covariances[t] = \
            kf.filter_update(filtered_state_means[t-1], filtered_state_covariances[t-1], measurements[t])
```

And for numerically robust "square root" filters

```python
from pykalman.sqrt import CholeskyKalmanFilter, AdditiveUnscentedKalmanFilter
kf = CholeskyKalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
ukf = AdditiveUnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, observation_covariance=0.1)
```

## Examples

Examples of all of `pykalman`'s functionality can be found in the scripts in
the `examples/` folder.
