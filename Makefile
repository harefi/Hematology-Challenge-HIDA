# Makefile ────────────────────────────────────────────────
.PHONY: setup train resume eda lint

ENV_NAME ?= hematol-22
DATA_ROOT ?= ./Source
OUTDIR ?= results/baseline_$(shell date +%Y%m%d_%H%M%S)

setup:              ## create/update Conda env
	conda env create -f environment.yml || conda env update -f environment.yml

train:              ## fresh baseline training
	python -m hematol.train --data_root $(DATA_ROOT) --output_dir $(OUTDIR)

resume:             ## resume from last_model.pth in the given OUTDIR
	python -m hematol.train --data_root $(DATA_ROOT) \
		--output_dir $(OUTDIR) --resume $(OUTDIR)/last_model.pth

eda:                ## open EDA notebook
	jupyter nbopen notebooks/00_eda.ipynb

lint:               ## run black + ruff
	black src/ scripts/
	ruff src/ scripts/
# ─────────────────────────────────────────────────────────

