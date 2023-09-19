install:
	pip install -e .[all]

upgrade:
	pip install --upgrade  -e .[all]

test:
	python -m pytest