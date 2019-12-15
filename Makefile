.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Build Cython cpp bindings
	CC=gcc-9 python setup.py build_ext --inplace

.PHONY: clean
clean: ## Remove development and build artifacts
	rm -f *.out
	rm -f .*.swp *.pyc
	rm -rf __pycache__
	rm -rf build/
	rm -f *.so
	rm -f py_ur_kin.cpp

.PHONY: test
test: build ## Build Cython bindings and execute tests
	pytest
