#!/usr/bin/make
# Copyright 2023 Scintillometry-Tools Contributors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# 	https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LIBNAME=scintillometry

ifeq (, $(shell which python ))
  $(error "PYTHON=$(PYTHON) not found in $(PATH)")
endif
PYTHON=$(shell which python)
PYTHON_VERSION=$(shell $(PYTHON) -c "import sys;\
	version='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));\
	sys.stdout.write(version)")
PYTHON_CHECK_MAJOR=$(shell $(PYTHON) -c 'import sys;\
  	print(int(float("%d"% sys.version_info.major) >= 3))')
PYTHON_CHECK_MINOR=$(shell $(PYTHON) -c 'import sys;\
  	print(int(float("%d"% sys.version_info.minor) >= 8))' )
PYTHON_SOURCES=src/$(LIBNAME)/[a-z]*.py

TEST_SOURCES=tests/[a-z]*.py
DOCS_SOURCES=docs

LOG_DATE=$(shell date +%Y%m%d_%H%M%S)

.PHONY:	help
help:	## Display this help screen
		@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY:	install
install:	--install-deps	## Install base package with conda/mamba
		@echo "\nInstalling editable..."
		@pip install -e .

install-tests:	--install-deps	## Install with dependencies for tests
		@echo "\nInstalling dependencies (tests)..."
		@$(pkg-manager) install "pytest>=7.0" "pytest-dependency>=0.5" "coverage>=7.1" \
		-c conda-forge -y
		@echo "\nInstalling editable with tests..."
		@pip install -e .[tests]

install-docs:	--install-deps	## Install with local documentation
		@echo "\nInstalling dependencies (docs)..."
		@$(pkg-manager) install sphinx "sphinx-rtd-theme>=1.1" -c conda-forge -y
		@echo "\nInstalling editable with documentation..."
		@make docs

install-all:	--install-deps	## Install package with tests & documentation
		@echo "\nInstalling dependencies (tests, docs)..."
		@$(pkg-manager) install "pytest>=7.0" "pytest-dependency>=0.5" "coverage>=7.1" \
		sphinx "sphinx-rtd-theme>=1.1" -c conda-forge -y
		@echo "\nInstalling editable with tests & documentation..."
		@pip install -e .[tests,docs]
		@make docs

install-dev:	--check-python --hook-manager	## Install in development mode
		@echo "\nInstalling dependencies (dev)..."
		@$(pkg-manager) install --file dev-requirements.txt -c conda-forge
		@echo "\nInstalling editable in development mode..."
		@pip install -e .[dev]

.PHONY:	tests
tests:	format flake8 coverage pylint	## Format code and run tests

.PHONY:	commit
commit:	tests	## Format, test, then commit
		@echo "\nCommitting..."
		@git commit

.PHONY: docs
docs:	## Build documentation
		@echo "\nBuilding documentation..."
		@cd $(DOCS_SOURCES); make clean && make html

format:	isort black	## Format all python files

commands:	## Display help for scintillometry package
		@$(PYTHON) ./src/$(LIBNAME)/main.py -h

run:	commands	## Alias for `make commands`

flake8:	## Lint with flake8
		@echo "\nLinting with flake8..."
		@flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
		@flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

.PHONY:	coverage
coverage:	## Run pytest with coverage
		@echo "\nRunning tests..."
		@mkdir -p "./logs/coverage"
		@coverage run --rcfile .coveragerc -m pytest && coverage html

pylint:	## Lint with Pylint
		@echo "\nLinting with pylint..."
		@pylint --rcfile .pylintrc **/*.py

black:	## Format all python files with black
		@echo "\nFormatting..."
		@black $(PYTHON_SOURCES)
		@black $(TEST_SOURCES)

isort:	## Optimise python imports
		@isort $(PYTHON_SOURCES)
		@isort $(TEST_SOURCES)

scalene:	## Profile with scalene (Python 3.9+)
		@echo "\nProfiling with Scalene..."
		@mkdir -p "./logs/scalene_logs"
		@scalene --cli --json \
		--outfile "./logs/scalene_logs/profile_$(LOG_DATE).json" \
		--- ./src/$(LIBNAME)/main.py -i "./tests/test_data/test_fake_bls_data.mnd" \
		-p "./tests/test_data/test_fake_path_transect.csv" \
		-t "CET" -v

.PHONY:	pkg
pkg:	tests docs --build	## Run test, build documentation, build package

--install-deps:	--check-python --hook-manager	# Private: install core dependencies only
		@echo "\nInstalling dependencies (core)..."
		@$(pkg-manager) install --file requirements.txt -c conda-forge

--check-python:	# Private: check Python is >=3.8
ifeq ($(PYTHON_CHECK_MAJOR),0)
	$(error "Python version is $(PYTHON_VERSION). Requires Python >= 3.8")
else ifeq ($(PYTHON_CHECK_MINOR),0)
	$(error "Python version is $(PYTHON_VERSION). Requires Python >= 3.8")
endif

--hook-manager:	# Private: hook package manager
ifneq (,$(findstring mamba, ${CONDA_EXE}))
pkg-manager := @mamba
else ifeq (,$(findstring conda, ${CONDA_EXE}))
pkg-manager := @conda
else
	$(error "No conda/mamba installation found. Try pip install -e . instead")
endif

--build:	# Private: build scintillometry package
		$(PYTHON) -m build
