.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Build Cython cpp bindings
	python setup.py build_ext --inplace

.PHONY: clean
clean: ## Remove development and build artifacts
	rm -f .*.swp *.pyc
	rm -rf __pycache__

.PHONY: test
test: build ## Build Cython bindings and execute tests
	pytest
