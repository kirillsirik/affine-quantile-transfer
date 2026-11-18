# Commands. Run them from the repository root.
#   make setup       create .venv with the pinned dependencies (needs Python 3.12; set PYTHON=...)
#   make check       unit tests, checks of the saved results, and every number of the paper against them
#   make smoke       small runs of every study, compared with the saved rows of the same seeds
#   make reproduce   full runs of every study, compared with the saved results, and every number of the paper
#   make figures     redraw the three figures of the paper and compare them with figures/reference/
#   make acs-data    download the Census files and rebuild the ACS extracts into build/ (needs the network)
#   make clean       remove build/
# Variables: PYTHON (default python3.12), WORKERS for make reproduce (default 4), and the output
# directories SMOKE_DIR, REPRODUCE_DIR and FIGURES_DIR under build/.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:
MAKEFLAGS += --no-builtin-rules

PYTHON ?= python3.12
VENV ?= .venv
PY := $(VENV)/bin/python
WORKERS ?= 4
BUILD ?= build
SMOKE_DIR ?= $(BUILD)/smoke
REPRODUCE_DIR ?= $(BUILD)/reproduce
FIGURES_DIR ?= $(BUILD)/figures

export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
export PYTHONDONTWRITEBYTECODE := 1

INSTALLED := $(VENV)/.installed
RUNNER := scripts/run_study.py scripts/environment.py
MQ_SOURCES := $(wildcard multiquantile/*.py)
GRID_SOURCES := singlequantile/grid/run_simulations.py
SINGLE_SOURCES := $(wildcard singlequantile/*.py)
CLOSED_FORM_SOURCES := $(wildcard closedform/*.py)
ACS_SOURCES := $(wildcard acs/*.py)
STUDIES := grid single quartiles slope-test closed-form acs

.PHONY: help setup test check smoke reproduce figures acs-data clean

help:
	@sed -n 's/^#   //p' $(firstword $(MAKEFILE_LIST))

setup: $(INSTALLED)

$(INSTALLED): requirements.txt
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Python 3.12 is required, found " + sys.version.split()[0] + "; set PYTHON=/path/to/python3.12")'
	$(PYTHON) -m venv --clear $(VENV)
	$(PY) -m pip install --disable-pip-version-check --no-input --quiet --require-hashes --only-binary :all: -r requirements.txt
	$(PY) -m pip check
	$(PY) -m scripts.environment --out $(VENV)/environment.json
	touch $@

test: $(INSTALLED)
	$(PY) -m unittest discover --start-directory tests --top-level-directory .

check: test
	$(PY) -m scripts.check_saved_results
	$(PY) -m scripts.check_numbers

smoke: $(INSTALLED)
	rm -rf $(SMOKE_DIR)
	$(PY) -m scripts.run_study all --mode smoke --out $(SMOKE_DIR) --workers 1
	$(PY) -m scripts.environment --out $(SMOKE_DIR)/environment.json --workers 1
	$(PY) -m scripts.compare_runs --mode smoke --run-dir $(SMOKE_DIR) --json $(SMOKE_DIR)/comparison.json

reproduce: $(foreach s,$(STUDIES),$(REPRODUCE_DIR)/$(s)/run.json) | $(INSTALLED)
	$(PY) -m scripts.environment --out $(REPRODUCE_DIR)/environment.json --workers $(WORKERS)
	$(PY) -m scripts.compare_runs --mode full --run-dir $(REPRODUCE_DIR) --json $(REPRODUCE_DIR)/comparison.json
	$(PY) -m scripts.check_numbers --run-dir $(REPRODUCE_DIR)

# A study is complete once its run.json exists; it reruns only if its code or configuration changed.
$(REPRODUCE_DIR)/grid/run.json: $(GRID_SOURCES) $(RUNNER) | $(INSTALLED)
	$(PY) -m scripts.run_study grid --mode full --out $(REPRODUCE_DIR) --workers $(WORKERS)

$(REPRODUCE_DIR)/quartiles/run.json: $(MQ_SOURCES) multiquantile/config/quartiles.json $(RUNNER) | $(INSTALLED)
	$(PY) -m scripts.run_study quartiles --mode full --out $(REPRODUCE_DIR) --workers $(WORKERS)

$(REPRODUCE_DIR)/single/run.json: $(SINGLE_SOURCES) singlequantile/config/single.json $(GRID_SOURCES) $(RUNNER) | $(INSTALLED)
	$(PY) -m scripts.run_study single --mode full --out $(REPRODUCE_DIR) --workers $(WORKERS)

$(REPRODUCE_DIR)/slope-test/run.json: $(MQ_SOURCES) multiquantile/config/slope_test.json $(RUNNER) | $(INSTALLED)
	$(PY) -m scripts.run_study slope-test --mode full --out $(REPRODUCE_DIR) --workers $(WORKERS)

$(REPRODUCE_DIR)/closed-form/run.json: $(CLOSED_FORM_SOURCES) multiquantile/weighting.py $(RUNNER) | $(INSTALLED)
	$(PY) -m scripts.run_study closed-form --mode full --out $(REPRODUCE_DIR) --workers $(WORKERS)

$(REPRODUCE_DIR)/acs/run.json: $(ACS_SOURCES) acs/config/acs.json $(wildcard acs/data/*) $(MQ_SOURCES) $(RUNNER) | $(INSTALLED)
	$(PY) -m scripts.run_study acs --mode full --out $(REPRODUCE_DIR) --workers $(WORKERS)

figures: $(INSTALLED)
	$(PY) -m scripts.make_figures --out $(FIGURES_DIR)

acs-data: $(INSTALLED)
	$(PY) -m acs.build_data --raw $(BUILD)/acs-raw --out $(BUILD)/acs-data --check

clean:
	rm -rf $(BUILD)
