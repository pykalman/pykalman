"""Tests to run without pytest, to check pytest isolation."""

from skbase import all_objects

# all_objects crawls all modules excepting pytest test files
# if it encounters an unisolated import, it will throw an exception
results = all_objects()
