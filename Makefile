PHONY: dev-fix

dev-fix:
	python -m black *.py
	python -m ruff check --fix *.py
	

