# bearing-order-soft-constraint

Code and per-seed results for the paper

> **Fault-Characteristic-Order Soft Constraints for Few-Shot Bearing Fault Diagnosis:
> Speed-Precision Insensitivity and the Identifiability Limit of Tachometer-Free Estimation**

---

## What this repository contains

A fault-characteristic-**order** soft-constraint loss `L_order` for few-shot bearing fault
diagnosis, together with a tachometer-free rotation-frequency estimator, and the full set of
controls used in the paper: a randomised-frequency control, a four-arm comparison of
rotation-frequency sources, a `lambda_max` dose-response sweep, a support/query overlap
control, a loss-definition control (masking NC out of the constraint), and cross-dataset
replications on PHM09 and on the HUST bearing dataset.

### Honest summary of the results

This repository reproduces **both** the positive and the negative results reported in the
paper. In short:

- On CWRU at −12 dB, the constraint yields **+1.11 pp (cross-2 hp)** and **+1.24 pp
  (cross-3 hp)** over the unconstrained baseline, significant after Bonferroni correction —
  **but this is conditional on the loss convention**: masking the (admittedly erroneous)
  supervision of NC samples out of `L_order` reduces the gain to +0.59 / +0.34 pp,
  not significant after correction (see `--mask-nc`).
- On PHM09 the cross-speed replication yields **no gain**, and the failure is traced to the
  fault-order lines not being discriminative in a geared transmission.
- On the HUST bearing dataset the identifiability condition *is* satisfied, yet there is still
  **no gain** (−2.73 pp), and the true-order prior is statistically indistinguishable from a
  randomised one. Identifiability is therefore a **necessary but not sufficient** condition.
- A zero-training probe of the prior's own discriminability does **not** separate the datasets
  cleanly, so no simple quantitative boundary rule is claimed.

---

## Repository layout

```
code/                     # all source
  models/                 # 1D-CNN backbone (<50K parameters)
  losses/                 # L_order, lambda scheduler, tachometer-free estimator
  data/                   # CWRU loader, windowing, noise injection, peak extraction
  training/               # evaluate.py / train.py (the main entry point)
  configs/default.yaml    # all hyper-parameters
  tests/                  # pytest suite
  deployment/             # INT8 quantisation / ESP32-S3 export chain
results/
  tables/                 # aggregate tables (CSV) + the per-experiment JSON summaries
  per_seed/               # one result.json + scheduler.json per (tag, method, seed)
```

## Quick start

```bash
python -m venv venv
# Windows / CUDA 12.x (Blackwell needs cu128 and torch>=2.7):
venv/Scripts/python -m pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/cu128
venv/Scripts/python -m pip install -r code/requirements.txt
venv/Scripts/python -m pytest tests -q          # unit tests (run from code/)
```

All commands in this README must be run with `code/` as the working directory
(the package is `training.evaluate`).

## Reproducing the experiments

See **[REPRODUCE.md](REPRODUCE.md)** for the exact command line of every experiment and the
table it produces.

## Data

| Dataset | Source | In this repo? |
|---|---|---|
| CWRU 12 kHz drive-end | <https://engineering.case.edu/bearingdatacenter> | no — download and point `data.raw_dir` at it |
| PHM09 gearbox | <https://phmsociety.org/data-analysis-competition/> | no |
| HUST bearing (Thuan & Hong) | <https://doi.org/10.17632/cbv7jyx4p9.3> (Kaggle mirror: `mohdsufianbinothman/hust-bearing`) | no |

Window pools are **derived artefacts** and are not committed; they are rebuilt from the raw
files by the scripts in `code/data/`. This keeps the repository small and avoids re-hosting
third-party data.

## Licence

See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).
