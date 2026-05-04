# Synthetic Changepoint Dataset for Anomalous Diffusion

This repository documents the construction and validation of a synthetic dataset for changepoint localization in anomalous diffusion trajectories.

Each trajectory has fixed length `L = 100` and contains a single changepoint separating two segments generated from different diffusion models. The considered models are ATTM, CTRW, FBM, LW, and SBM.

## Dataset design

- Trajectory length: 100
- Minimum segment length: 20
- Ordered transitions: 5 × 4 = 20
- Training set: 20 × 10,000 = 200,000 trajectories
- Validation set: 20 × 1,000 = 20,000 trajectories
- Test set: 20 × 10,000 = 200,000 trajectories

## Repository structure

```text
notebooks/
└── 01_synthetic_dataset_changepoint_anomalous_diffusion.ipynb

scripts/
└── build_synthetic_changepoint_dataset.py

data_synthetic_changepoint_andi/
├── dataset_summary.json
├── train_pair_counts.csv
├── val_pair_counts.csv
└── test_pair_counts.csv