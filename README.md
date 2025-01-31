# Welcome to pykalman

<a href="https://github.com/pykalman/pykalman"><img src="https://github.com/pykalman/pykalman/blob/main/doc/source/images/pykalman-logo-with-name.png" width="175" align="right" /></a>

**the dead-simple Kalman Filter, Kalman Smoother, and EM library for Python.**

`pykalman` is a Python library for Kalman filtering and smoothing, providing efficient algorithms for state estimation in time series. It includes tools for linear dynamical systems, parameter estimation, and sequential data modeling. The library supports the Kalman Filter, Unscented Kalman Filter, and EM algorithm for parameter learning.

:rocket: **Version 0.10.0 out now!** [Check out the release notes here](https://github.com/pykalman/pykalman/blob/main/CHANGELOG.md).

|  | **[Documentation](https://pykalman.github.io/)** · **[Tutorials](https://github.com/pykalman/pykalman/tree/main/examples)** |
|---|---|
| **Open&#160;Source** | [![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/pykalman/pykalman/blob/main/LICENSE) |
| **Community** | [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen)](https://discord.com/invite/54ACzaFsn7) [![!LinkedI](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/scikit-time/) |
| **Code** | [![!pypi](https://img.shields.io/pypi/v/pykalman?color=orange)](https://pypi.org/project/pykalman/) [![!python-versions](https://img.shields.io/pypi/pyversions/pykalman)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |
| **Downloads** | ![PyPI - Downloads](https://img.shields.io/pypi/dw/pykalman) ![PyPI - Downloads](https://img.shields.io/pypi/dm/pykalman) [![Downloads](https://static.pepy.tech/personalized-badge/pykalman?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/pykalman) |


## :speech_balloon: Where to ask questions

Questions and feedback are extremely welcome! We strongly believe in the value of sharing help publicly, as it allows a wider audience to benefit from it.

| Type                            | Platforms                               |
| ------------------------------- | --------------------------------------- |
| :bug: **Bug Reports**              | [GitHub Issue Tracker]                  |
| :sparkles: **Feature Requests & Ideas** | [GitHub Issue Tracker]                       |
| :woman_technologist: **Usage Questions**          |  [Stack Overflow] |
| :speech_balloon: **General Discussion**        | [Discord] |
| :factory: **Contribution & Development** | `dev-chat` channel · [Discord] |
| :globe_with_meridians: **Meet-ups and collaboration sessions** | [Discord] - Fridays 13 UTC, dev/meet-ups channel |

[github issue tracker]: https://github.com/pyklaman/pykalman/issues
[stack overflow]: https://stackoverflow.com/questions/tagged/pykalman
[discord]: https://discord.com/invite/54ACzaFsn7


## :hourglass_flowing_sand: Install pykalman

- **Operating system**: macOS X · Linux · Windows 8.1 or higher
- **Python version**: Python 3.9, 3.10, 3.11, 3.12, and 3.13
- **Package managers**: [pip](https://pip.pypa.io/en/stable/)

For a quick installation::
```bash
pip install pykalman
```

Alternatively, you can setup from source:

```bash
pip install .
```

## :zap: Usage

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

Examples of all of `pykalman`'s functionality can be found [here](https://github.com/pykalman/pykalman/tree/main/examples).
