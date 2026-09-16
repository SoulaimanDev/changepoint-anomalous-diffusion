# Completed final training / validation-calibration phase — 25 runs

**These are validation-calibration results, NOT independent test results.** The backup dated 16 September 2026 contains 25/25 completed runs, 0 failed, 0 running and 0 missing. Each of the five selected architectures/configurations has seeds `1, 2, 3, 42, 123`.

The preceding [450-run tuning campaign](../tuning/README.md) used 30 configurations per architecture and seeds `11, 29, 47`. Its selected slots and published metrics remain unchanged. Each selected configuration was then trained with five final seeds; checkpoint selection used validation_tuning, followed by threshold calibration on validation_calibration. All 25 checkpoint/configuration/threshold seals are present and verified. The independent held-out test is a separate subsequent phase and is excluded from this publication.

## Five-seed validation-calibration summary

| Architecture | Configuration | Seeds | F1 mean ± std | MAE_all mean ± std | FPR mean ± std |
|---|---|---:|---:|---:|---:|
| LSTM | cfg_02 | 5 | 0.791694 ± 0.002988 | 9.215873 ± 0.079549 | 0.194040 ± 0.003057 |
| xLSTM | cfg_09 | 5 | 0.788195 ± 0.005167 | 9.690845 ± 0.184891 | 0.188920 ± 0.006128 |
| CNN-LSTM | cfg_06 | 5 | 0.786514 ± 0.003121 | 7.599934 ± 0.055970 | 0.186640 ± 0.003812 |
| Transformer | cfg_06 | 5 | 0.792640 ± 0.005474 | 7.640302 ± 0.166553 | 0.191920 ± 0.004621 |
| ConvTransformer | cfg_12 | 5 | 0.799563 ± 0.005949 | 7.294523 ± 0.238058 | 0.195240 ± 0.003971 |

Values show arithmetic mean ± sample standard deviation (`ddof=1`, five seeds). The CSVs preserve full exported precision; the table rounds to six decimal places. MAE_all is measured in trajectory-position units across all truly positive calibration trajectories, including detection false negatives. Standard deviations describe across-seed variability; they are not confidence intervals or statistical significance tests.

ConvTransformer obtained the highest observed mean validation-calibration F1 among the evaluated architectures under this experimental protocol. These measurements do not establish general superiority or independent-test generalization. Calibration data were used to select thresholds, so this is not an unbiased held-out performance estimate. Numerical MAE values are not directly comparable to papers reporting RMSE or different changepoint metrics.

## Metric origin and epoch convention

Every F1, FPR, MAE_all and frozen threshold in [final_25_runs.csv](final_25_runs.csv) comes from the corresponding archived `final_seal.json` (`calibration_metrics` and `threshold`). We do not substitute the different validation_tuning metrics in `run.json`, nor values from an intermediate epoch. `epochs_completed` is the length of `epochs.json`; `best_epoch` is the original **zero-based** `run.json/selected_epoch`, cross-checked with the final epoch record. No model was loaded, retrained or recalibrated for this export.

The frozen threshold is on the original 0.01 grid. Its full metric row matches the archived calibration grid selection (maximize F1 subject to FPR ≤ 0.20, then minimize FPR, then prefer the highest threshold). All 25 calibrated FPRs satisfy the cap. Thresholds were frozen before held-out testing; no independent-test artifacts were read for this update.

## Provenance and verification

- [source_provenance.json](source_provenance.json) records the backup digest, archived per-run/checkpoint/seal/configuration/epoch hashes, full sealed calibration metric rows and common environment. It contains no absolute machine paths.
- Backup SHA-256: `262643c9615d2695ebd437ce4291a33e2125d8c89f85a24a37ee7285e73d8089`.
- The later merged registry was not present in the backup. Its rows were reconstructed outside Git from the five archived registries using the archived CSV writer and the ordering/fields of the archived final merger. The SHA-256 exactly matches the independently supplied later-merge digest: `1b93c0309e7c20475fb36c2e85b3475cb73221f5043c646cb79783cea94bb475`. This registry is distinct from the metric CSV published here.
- The archive's complete code inventory and frozen configuration checks passed with code hash `a91e1ddc8c99c8398f679ed10f5f8b057b433f622378ab85d8f5cbaa136094c9`. All seal references, checkpoint bytes, run registry hashes, identities and configuration slots were checked against the extracted archive. No original artifact was edited.
- Environment: Python 3.11.9; PyTorch 2.11.0+cu128; CUDA 12.8; cuDNN 91900; NumPy 1.26.4; h5py 3.14.0; PyYAML 6.0.2; NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb.
- The owner identifies the backup as preceding independent-test access (`test_accessed=false`). This audit uses only final training/calibration artifacts and archived code; it does not inspect any ongoing or later test execution or independently certify its access history.

The public proof is a compact derived export. Its source hashes permit comparison with the verified archive; the archive, raw logs and checkpoints are intentionally not distributed here. It does not replace access to the full scientific package for reproduction of training.

## Read-only validation

From the repository root:

```bash
python -B scripts/summarize_final_validation_results.py
```

The standard-library script checks 25 unique completed identities, five seeds per architecture, agreement with the frozen tuning selections, exact agreement with exported sealed metrics, FPR/threshold constraints, independent recomputation of [architecture_summary.csv](architecture_summary.csv), and the Markdown table. It neither opens datasets/checkpoints nor writes files.
