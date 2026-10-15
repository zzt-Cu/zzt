# Reproducing every table

All CWRU experiments are driven by `training.evaluate`; run them with **`code/` as the working
directory**. Seeds are always `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`.
The per-seed records of every run below are included in `results/per_seed/`, and the aggregate
tables in `results/tables/`.

## Shared setup

```bash
cd code
python -m training.evaluate --help
```

## Table 4 — main result (3-way, −12 dB, 16 seeds)

```bash
python -m training.evaluate \
  --methods CNN-CE,CNN-Freq --classes NC,IR007,OR007 \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --snr-db -12 --n-shot 5 --epochs 80 --val-every 20 \
  --warmup-epochs 4 --lambda-step 2 --lambda-max 0.5 \
  --tag b_3c_s12_sh5_s8_s16b --device cuda
```
(seeds 0–7 were run under tag `b_3c_s12_sh5_s8`, seeds 8–15 under `b_3c_s12_sh5_s8_s16b`)

## Table 5 — randomised-frequency control

```bash
python -m training.evaluate ...same... --freq-jitter 0.5 --jitter-seed 0 \
  --tag b_3c_s12_sh5_jit_s16b
```

## Table 6 — effect of the fourth class (4-way)

```bash
python -m training.evaluate --methods CNN-CE,CNN-Freq --classes NC,IR007,OR007,B007 \
  --seeds ... --snr-db -12 --n-shot 5 --epochs 80 --val-every 20 \
  --warmup-epochs 4 --lambda-step 2 --lambda-max 0.5 \
  --tag b_4c_s12_sh5_s16b --device cuda
```

The IR014 prior-valid control (seeds 0–15) is:

```bash
python -m training.evaluate --methods CNN-CE,CNN-Freq \
  --classes NC,IR007,OR007,IR014 --seeds ... --snr-db -12 --n-shot 5 --epochs 80 \
  --val-every 20 --warmup-epochs 4 --lambda-step 2 --lambda-max 0.5 \
  --tag c4_ir014_s16 --device cuda
```

## Table 7 — four-arm comparison of rotation-frequency sources

```bash
# arm (1) measured/nominal
python -m training.evaluate ... --speed-mode fixed        --tag b_3c_s12_sh5_s8_s16b
# arm (2) deliberately wrong by +5.7 %
python -m training.evaluate ... --speed-mode fixed --fixed-rpm 1900 \
      --tag ord3c_s12_sh5_rpm1900_s16b
# arm (3) per-episode estimate (tachometer-free)
python -m training.evaluate ... --speed-mode estimate     --tag ord3c_s12_sh5_est_s16b
# arm (4) interval randomisation ±100 rpm
python -m training.evaluate ... --speed-mode jitter --speed-jitter-rpm 100 \
      --tag ord3c_s12_sh5_jit100_s16b
```

## Table 8 — `lambda_max` dose-response

```bash
python -m training.evaluate --methods CNN-Freq --classes NC,IR007,OR007 \
  --seeds ... --snr-db -12 --n-shot 5 --epochs 80 --val-every 20 \
  --warmup-epochs 4 --lambda-step 2 --lambda-max 0 --tag lamsweep_l0_s16
python -m training.evaluate ... --lambda-max 0.1 --tag lamsweep_l01_s16
# lambda_max = 0.5 is the main experiment (b_3c_s12_sh5_s8_s16b) and is reused here
```

## Table 9 — support/query overlap control

```bash
python -m training.evaluate --methods CNN-CE,CNN-Freq --classes NC,IR007,OR007 \
  --seeds ... --snr-db -12 --n-shot 5 --epochs 80 --val-every 20 \
  --warmup-epochs 4 --lambda-step 2 --lambda-max 0.5 \
  --disjoint-support-query --tag disjoint_s16 --device cuda
```

## Table 14 — loss-definition control (masking NC out of `L_order`)

```bash
# arm (2): mean over fault queries only
python -m training.evaluate --methods CNN-Freq --classes NC,IR007,OR007 \
  --seeds ... --snr-db -12 --n-shot 5 --epochs 80 --val-every 20 \
  --warmup-epochs 4 --lambda-step 2 --lambda-max 0.5 \
  --mask-nc --tag masknc3c_s16 --device cuda

# arm (3): same, with the gradient-scale confound removed
python -m training.evaluate ... --mask-nc --mask-nc-rescale \
  --tag maskncrs3c_s16 --device cuda
```

## PHM09 and HUST bearing

These two datasets are handled by standalone scripts that are **not** part of this package
(they use the same backbone and the same loss, but a different data pipeline). The per-seed
records of those runs are **not** included here; the aggregate results are in the paper and
their scripts are available from the corresponding author.

## Cross-checking a table without re-training

Every aggregate table in `results/tables/` is derived from the per-seed `result.json` files in
`results/per_seed/`. Recomputing a table is therefore a pure post-processing step; no GPU is
required.
