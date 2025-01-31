# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.10.0] - 2025-01-31

### Maintenance

- [MNT] Clean-up of `pyproject.toml` ([#130](https://github.com/pykalman/pykalman/pull/130)) [@phoeenniixx](https://github.com/phoeenniixx)
- [MNT] CI element for programmatic check of missing `__init__` files ([#131](https://github.com/pykalman/pykalman/pull/131)) [@phoeenniixx](https://github.com/phoeenniixx)
- [MNT] fix `pykalman` logo in `README.md` ([#134](https://github.com/pykalman/pykalman/pull/134)) [@fkiraly](https://github.com/fkiraly)
- [MNT] adding CI test matrix ([#127](https://github.com/pykalman/pykalman/pull/127)) [@phoeenniixx](https://github.com/phoeenniixx)
- [MNT] temporarily remove missing init file check ([#125](https://github.com/pykalman/pykalman/pull/125)) [@fkiraly](https://github.com/fkiraly)
- by [@KOLANICH](https://github.com/KOLANICH) [MNT] Move the metadata into PEP-621-compliant pyproject.toml ([#124](https://github.com/pykalman/pykalman/pull/124)) [@KOLANICH](https://github.com/KOLANICH), [@fkiraly](https://github.com/fkiraly)
- [MNT] python 3.11 compatibility - replace deprecated inspect.getargspec ([#107](https://github.com/pykalman/pykalman/pull/107)), ([#101](https://github.com/pykalman/pykalman/pull/101)), ([#63](https://github.com/pykalman/pykalman/pull/63)) [@doberbauer](https://github.com/doberbauer), [@erkinsahin](https://github.com/erkinsahin), [@jonathanng](https://github.com/jonathanng)
- [MNT] add all-contributors file ([#123](https://github.com/pykalman/pykalman/pull/123)) [@fkiraly](https://github.com/fkiraly)
- [MNT] Added pre-commit Hooks for code-quality checks ([#117](https://github.com/pykalman/pykalman/pull/117)) [@phoeenniixx](https://github.com/phoeenniixx)

### Documentation

- [DOC] `pykalman` logo with name ([#136](https://github.com/pykalman/pykalman/pull/136))  [@fkiraly](https://github.com/fkiraly)
- [DOC] Update `README.md `([#132](https://github.com/pykalman/pykalman/pull/132)) [@phoeenniixx](https://github.com/phoeenniixx)
- [DOC] improved formatting of readme preamble, new pykalman logo ([#133](https://github.com/pykalman/pykalman/pull/133)) [@fkiraly](https://github.com/fkiraly)
- [DOC] Fix notation wrt observation offsets ([#70](https://github.com/pykalman/pykalman/pull/70)) [@dslaw](https://github.com/dslaw)
- [DOC] fix simple typo, probabily -> probably ([#96](https://github.com/pykalman/pykalman/pull/96)) [@timgates42](https://github.com/timgates42)

### Enhancements

- [ENH] Fixed variable and argument names `_em_observation_covariance` ([#69](https://github.com/pykalman/pykalman/pull/69)) [@oseiskar](https://github.com/oseiskar)
- [ENH] merge of `sktime` fork into `pykalman` - `numpy 2` compatibility, documentation, test refactor ([#120](https://github.com/pykalman/pykalman/pull/120)) [@fkiraly](https://github.com/fkiraly)

### Contributors
[@doberbauer](https://github.com/doberbauer), [@dslaw](https://github.com/dslaw), [@erkinsahin](https://github.com/erkinsahin), [@fkiraly](https://github.com/fkiraly), [@jonathanng](https://github.com/jonathanng), [@KOLANICH](https://github.com/KOLANICH), [@oseiskar](https://github.com/oseiskar), [@phoeenniixx](https://github.com/phoeenniixx), [@timgates42](https://github.com/timgates42)

## [v0.9.7] - 2023-09-19
### Fixed
- `Masked array are not supported` in `linalg.solve` fixed by migration to `numpy.linalg.solve`
- Python 3.11 compatibility issue `inspect.getargspec` -> `inspect.getfullargspec` resolved

### Changed
- Migrated from `setuptools` format to `pyproject.toml`

[Unreleased]: https://github.com/pykalman/pykalman/compare/v0.9.7...HEAD
[v0.9.7]: https://github.com/pykalman/pykalman/compare/92810c9e3005dde7d9fe063607e4c96861d087d4...v0.9.7
