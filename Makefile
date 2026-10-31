.PHONY: test demo serve clean

PYTHONPATH := src

test:
	PYTHONPATH=$(PYTHONPATH) python3 -m pytest -q

demo:
	PYTHONPATH=$(PYTHONPATH) python3 -m explainability.cli --out-dir reports

serve:
	PYTHONPATH=$(PYTHONPATH) python3 -m explainability.cli --serve

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf reports .pytest_cache
