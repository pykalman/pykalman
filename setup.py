from setuptools import setup, find_packages

setup(
    name = 'pykalman',
    version = '0.9.5',
    author = 'Daniel Duckworth',
    author_email = 'pykalman@gmail.com',
    description = ('An implementation of the Kalman Filter, Kalman ' +
      'Smoother, and EM algorithm in Python'),
    license = 'BSD',
    keywords = 'kalman filter smoothing em hmm tracking unscented ukf kf',
    url = 'http://pykalman.github.com',
    packages = find_packages(),
    package_data={'pykalman': ['datasets/descr/robot.rst', 'datasets/data/robot.mat']},
    classifiers = [
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: BSD License',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    include_package_data = True,
    install_requires = [
      'numpy',
      'scipy',
    ],
    tests_require = [
      'nose',
    ],
    extras_require = {
        'docs': [
          'Sphinx',
          'numpydoc',
        ],
        'tests': [
          'nose',
        ],
    },
)
