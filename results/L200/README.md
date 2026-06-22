# ConvTransformer L=200 - final GLOBAL results

These artifacts were extracted from `notebooks/13_convtransformer_L200_detection_localization.ipynb` so the final thesis results can be inspected directly on GitHub.

## Run provenance

- Mode: `GLOBAL_RUN`
- Train / validation / test: 200,000 / 20,000 / 200,000 trajectories
- Trajectory length: 200 (`dx` length 199)
- Seed: 42
- Maximum epochs: 100
- Epochs completed: 61 (early stopping)
- Selected validation threshold: 0.475

## Main test metrics

| Accuracy | Precision | Recall | F1 | FPR | FNR | MAE all | RMSE all |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.852365 | 0.894449 | 0.799020 | 0.844046 | 0.094290 | 0.200980 | 9.376101 | 18.274097 |

`global_test_summary.csv` is the authoritative compact metric table. Threshold tables, error-by-position results, false-positive and false-negative breakdowns, and selected transition rankings are also included. The `figures/` directory contains the 19 PNG figures embedded in the executed notebook.

The source notebook uses a positive detection-class weight of 1.5, a negative weight of 1.0, and multitask loss weights `[2.0, 1.0]`. These details are part of the executed configuration and should be retained when interpreting the results.
