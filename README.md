# Hematology Image Classification 🩸🔬 (Helmholtz HIDA Challenge 2022)

[![CI](https://github.com/harefi/Hematology-Challenge-HIDA/actions/workflows/ci.yml/badge.svg)](https://github.com/harefi/Hematology-Challenge-HIDA/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Certificate](https://img.shields.io/badge/view-certificate-blue)](docs/certificate.pdf)

> **Project purpose:** demonstrate an end‑to‑end, reproducible computer‑vision pipeline as a portfolio piece — not to win the original competition.

[🔗 Official challenge page](https://www.helmholtz-hida.de/en/events/data-challenge-help-a-hematologist-out/)

---

## 🚀 30‑second Quick Start
```bash
# clone & set up env ─────────────────────────────
 git clone https://github.com/harefi/Hematology-Challenge-HIDA.git
 cd hematology-challenge
 conda env create -f environment.yml      # or: pip install -e .[dev]

# train baseline ResNet‑18 (timestamped output dir)
 make train

# live curves
 tensorboard --logdir results/*/tb --port 6006
```

---

## 📑 Table of Contents
1. [Background](#1-background)
2. [Data Download](#2-data-download)
3. [Environment](#3-environment)
4. [Repo Layout](#4-repo-layout)
5. [EDA Highlights](#5-eda-highlights)
6. [Training Pipeline](#6-training-pipeline)
7. [Baseline Metrics](#7-baseline-metrics)
8. [Road‑map](#8-road-map)
9. [References](#9-references)

---

## 1  Background
The dataset contains ≈33 k microscope blood‑cell images labelled into 11 morphological classes.  The hidden test server is offline, so validation metrics are computed with an 80 / 20 stratified split.

Goals for this repo:
* showcase **clean project structure, CI, tests, and documentation**;
* provide a solid **baseline** with reproducible logs / checkpoints;
* outline clear next steps for prospective employers to gauge thought process.

---

## 2  Data Download
Place the two original archives in `Source/`:
```
Source/
 ├─ Acevedo_20.zip
 └─ Matek_19.zip
```
`hematol.train` automatically extracts them into `Source/Acevedo_20/` & `Source/Matek_19/` on first run.

---

## 3  Environment
`environment.yml` targets **Python 3.11** + CUDA 12.  Install via Conda *or* editable pip:
```bash
conda env create -f environment.yml
conda activate hematol-22
# optional dev extras
pip install -e .[dev]
```

---

## 4  Repo Layout
```
├── src/hematol/             # 📦 importable package
│   ├── data.py              # dataset builders
│   ├── transforms.py        # aug & normalisation
│   ├── models.py            # backbone registry
│   └── train.py             # main training loop
├── scripts/                 # thin CLI shims (legacy)
├── notebooks/00_eda.ipynb   # lightweight, view‑only EDA
├── tests/                   # pytest sanity tests
├── results/                 # logs/, models/, tb/  (git‑ignored)
├── docs/certificate.pdf     # participation cert
├── Makefile                 # one‑liner tasks
└── .github/workflows/ci.yml # black + ruff + pytest
```

---

## 5  EDA Highlights
Key findings (see notebook):
* Moderate **class imbalance** – rarest class has ≈⅛ the samples of the commonest.
* Image resolutions vary between 300‑480 px; we resize → crop 224².
* Colour histograms similar across datasets ⇒ stain normalisation not mandatory for baseline.

---

## 6  Training Pipeline  `make train`
* **Data module** — on‑the‑fly resize 256 ⇒ random crop 224, flip, ±10° rotation.
* **Model** — ImageNet‑pretrained **ResNet‑18**; FC layer swapped for 11‑class output.
* **Optimiser** — AdamW 1e‑4; cosine LR warm‑up (TBD).
* **Logging** — CSV, TensorBoard, checkpoints every epoch (`last_model.pth`) + best by macro‑F1.
* **Resume** — `make resume OUTDIR=results/baseline_… epochs=50` continues from `last_model.pth`.

---

## 7  Baseline Metrics
| Split | Macro‑F1 | Accuracy | Epochs |
|-------|---------|----------|--------|
| Val   | **0.83** | 0.88 | 30 |

*(Stratified 80 / 20 split; run ID `baseline_2025‑05‑31`)*

---

## 8  Road‑map
* **Class‑weighted loss / focal‑loss** to address imbalance.
* **EfficientNet‑V2 & ConvNeXt‑T** fine‑tuning (expected +3‑5 pp F1).
* **Self‑supervised pre‑train** (BYOL) on unlabelled set.
* **Ensemble** top 3 checkpoints for robustness.

Pull‑requests welcome — open an issue to discuss ideas!

---

## 9  References
* [Helmholtz AI. *Data Challenge – Help a Hematologist Out* (2022).](https://www.helmholtz-hida.de/en/events/data-challenge-help-a-hematologist-out/)
* Matek, C., Schwarz, S., Spiekermann, K. et al. Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks. Nat Mach Intell 1, 538–544 (2019). https://doi.org/10.1038/s42256-019-0101-9
* Rupapara, V., Rustam, F., Aljedaani, W. et al. Blood cancer prediction using leukemia microarray gene data and hybrid logistic vector trees model. Sci Rep 12, 1000 (2022). https://doi.org/10.1038/s41598-022-04835-6
* Nazari E, Farzin AH, Aghemiri M, Avan A, Tara M, Tabesh H. Deep Learning for Acute Myeloid Leukemia Diagnosis. J Med Life. 2020 Jul-Sep;13(3):382-387. https://doi.org/10.25122/jml-2019-0090
*  Shaheen, Maneela, Khan, Rafiullah, Biswal, R. R., Ullah, Mohib, Khan, Atif, Uddin, M. Irfan, Zareei, Mahdi, Waheed, Abdul, Acute Myeloid Leukemia (AML) Detection Using AlexNet Model, Complexity, 2021, 6658192, 8 pages, 2021. https://doi.org/10.1155/2021/6658192 
