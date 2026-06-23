# Changepoint Detection in Synthetic Anomalous Diffusion Trajectories

This repository contains the experimental code for a Master's thesis on changepoint detection and temporal localization in synthetic anomalous diffusion trajectories.

The project studies trajectories generated from five anomalous diffusion models:

- Annealed Transient Time Motion (`ATTM`);
- Continuous Time Random Walk (`CTRW`);
- Fractional Brownian Motion (`FBM`);
- Lévy Walk (`LW`);
- Scaled Brownian Motion (`SBM`).

The main task is to detect whether a trajectory contains a changepoint and, when present, estimate its temporal position. The current frozen version uses normalized increments `dx(t)` as input.

## Experimental phases

### Phase 1: localization with guaranteed changepoint

All trajectories contain one changepoint. The model only has to estimate its temporal position.

### Phase 2: binary detection and localization

The dataset contains trajectories with and without changepoints. The model has two outputs:

1. a binary changepoint probability;
2. a discrete temporal localization distribution.

The four v2 architectures evaluated for `L=100` are:

- LSTM;
- CNN-LSTM;
- Transformer;
- ConvTransformer-v2.

The original notebooks keep the filename `convlstm`, but the implemented architecture is a CNN-LSTM: one-dimensional convolutional blocks followed by LSTM layers, not a ConvLSTM recurrent cell.

## Repository structure

```text
changepoint-anomalous-diffusion/
|
|-- configs/
|   |-- frozen_v2.yaml
|
|-- data_synthetic_changepoint_andi/
|   |-- dataset_summary.json
|   |-- train_pair_counts.csv
|   |-- val_pair_counts.csv
|   |-- test_pair_counts.csv
|
|-- docs/
|   |-- experiment_protocol_v2.md
|
|-- notebooks/
|   |-- 01_synthetic_dataset_changepoint_anomalous_diffusion.ipynb
|   |-- 02_lstm_changepoint_detection.ipynb
|   |-- 03_convlstm_changepoint_detection.ipynb
|   |-- 04_transformer_changepoint_detection.ipynb
|   |-- 05_convtransformer_changepoint_detection.ipynb
|   |-- 06_model_validation_changepoint_comparison.ipynb
|   |-- 07_synthetic_dataset_binary_changepoint.ipynb
|   |-- 08_lstm_binary_detection_localization.ipynb
|   |-- 09_convlstm_binary_detection_localization.ipynb
|   |-- 10_transformer_binary_detection_localization.ipynb
|   |-- 11_convtransformer_binary_detection_localization.ipynb
|
|-- results/
|   |-- frozen_v2/
|
|-- scripts/
|   |-- build_synthetic_changepoint_dataset.py
|
|-- README.md
```

## Frozen v2 baseline

The current results are frozen in `results/frozen_v2/` before adding ConvTransformer-v3, multi-seed repetitions, or a classical PELT baseline.

This folder is the reference baseline for future experiments. New results should be written to separate folders and should not overwrite the frozen v2 files.

Important interpretation notes:

- the v2 results correspond to one main execution per architecture, not to a multi-seed study;
- ConvTransformer-v2 is the best observed model in the current Phase 2 comparison;
- the difference between ConvTransformer-v2 and CNN-LSTM is small and should be interpreted cautiously until multi-seed results are available;
- the ConvTransformer `L=100` versus `L=200` comparison is stored for traceability, but it is descriptive because some training rules differ between the two runs.

See:

- `docs/experiment_protocol_v2.md`;
- `results/frozen_v2/run_manifest.json`;
- `results/frozen_v2/global_detection_metrics_L100.csv`;
- `results/frozen_v2/localization_metrics_L100.csv`.

## Phase 2 v2 results for L=100

| Model | Threshold | Accuracy | Precision | Recall | F1-score | FPR | FNR | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.35 | 0.732765 | 0.734069 | 0.729980 | 0.732019 | 0.264450 | 0.270020 | 12.733092 | 16.350369 |
| CNN-LSTM | 0.25 | 0.789130 | 0.797980 | 0.774280 | 0.785951 | 0.196020 | 0.225720 | 7.664480 | 11.918347 |
| Transformer | 0.30 | 0.771165 | 0.809284 | 0.709540 | 0.756137 | 0.167210 | 0.290460 | 8.741436 | 12.530686 |
| ConvTransformer-v2 | 0.30 | 0.803675 | 0.818374 | 0.780590 | 0.799036 | 0.173240 | 0.219410 | 7.536969 | 11.570613 |

ConvTransformer-v2 has the best observed F1-score and the lowest localization error in this single main execution. However, the localization difference with CNN-LSTM is small and is not interpreted as conclusive.

## Final thesis sources

The final LaTeX sources are stored in `docs/thesis/`. They include:

- the revised academic manuscript and bibliography;
- the explicit link and citation for this repository;
- the five-seed ConvTransformer-v2 robustness analysis;
- the full-test PELT baseline;
- a compact synthetic-data generation pseudocode;
- four qualitative ConvTransformer-v2 examples;
- a cautious exploratory discussion of ConvTransformer-v3a and v3b.

ConvTransformer-v3a and v3b were evaluated only through preliminary validation runs. They are not presented as replacements for ConvTransformer-v2, and no v3 test metrics are reported.

## How to run the current notebooks

Install the main dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow tqdm h5py
```

Run the notebooks in numerical order:

1. `notebooks/01_synthetic_dataset_changepoint_anomalous_diffusion.ipynb`
2. `notebooks/02_lstm_changepoint_detection.ipynb`
3. `notebooks/03_convlstm_changepoint_detection.ipynb`
4. `notebooks/04_transformer_changepoint_detection.ipynb`
5. `notebooks/05_convtransformer_changepoint_detection.ipynb`
6. `notebooks/06_model_validation_changepoint_comparison.ipynb`
7. `notebooks/07_synthetic_dataset_binary_changepoint.ipynb`
8. `notebooks/08_lstm_binary_detection_localization.ipynb`
9. `notebooks/09_convlstm_binary_detection_localization.ipynb`
10. `notebooks/10_transformer_binary_detection_localization.ipynb`
11. `notebooks/11_convtransformer_binary_detection_localization.ipynb`

Dataset paths may need to be adapted depending on the local execution environment.

## Completed and planned extensions

Completed:

1. ConvTransformer-v3a multi-channel input pipeline.
2. ConvTransformer-v3b soft-label localization pipeline.
3. Five-seed ConvTransformer-v2 robustness study under a fixed reduced-training protocol.
4. PELT baseline with validation-only penalty selection and full-test evaluation.

Still planned:

1. Full-training multi-seed repetitions.
2. Validation-only completion of the v3 ablation before any final test evaluation.
3. Controlled `L=100` versus `L=200` comparison with identical training rules.
4. Comparison with a diffusion-specific method under an identical protocol.

These extensions are not presented as an attempt to directly outperform AnDi methods. The objective is to build a more robust, reproducible and AnDi-inspired version under a clearly documented protocol.
