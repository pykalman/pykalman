install:
	pip install -e .[all]

upgrade:
	pip install --upgrade  -e .[all]

test:
	python -m pytest

install-build-deps:
	pip install setuptools wheel build toml

package:
	python -m build .

release: package
	twine upload dist/*
