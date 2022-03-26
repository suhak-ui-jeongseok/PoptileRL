.PHONY: init format check requirements

init:
	python3 -m pip install -U pipenv setuptools
	python3 -m pipenv install --dev

format:
	isort --profile black -l 119 poptile_rl
	black -S -l 119 poptile_rl

check:
	isort --check-only --profile black -l 119 poptile_rl
	black -S -l 119 --check poptile_rl
	python3 lint.py

requirements:
	python3 -m pipenv lock -r > requirements.txt
	python3 -m pipenv lock -dr > requirements-dev.txt
