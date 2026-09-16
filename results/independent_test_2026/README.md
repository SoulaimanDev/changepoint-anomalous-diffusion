# Independent held-out test — 25 frozen models

The verified archive dated **16 September 2026** contains exactly **25 completed test JSONs**, with no missing or duplicate identities: five architectures and seeds `1, 2, 3, 42, 123` for each. The selected configurations remain LSTM `cfg_02`, xLSTM `cfg_09`, CNN-LSTM `cfg_06`, Transformer `cfg_06` and ConvTransformer `cfg_12`.

## Protocol and separation of phases

1. [Tuning](../tuning/README.md): 30 configurations per architecture, seeds `11, 29, 47`, 450 runs; selection used validation only.
2. [Final training / validation-calibration](../final_validation_2026/README.md): five final seeds per selected configuration, checkpoint selection and threshold calibration before test. Those published artifacts are unchanged.
3. **Independent held-out test (this directory):** previously frozen models and thresholds evaluated on the held-out split. No training, calibration, model loading or dataset access was performed to prepare this publication.

For each test JSON, its recorded final-seal SHA-256 matches the previously published calibration provenance, and its threshold exactly matches that seal's calibration threshold (25/25). The archived authorization records `approved_test=true`. These checks establish consistency with the archived pre-test seals; this publication does not rerun or independently observe the original evaluation process. The test is not used to change configurations, checkpoints, thresholds or any earlier decision.

Each result's confusion counts cover 200,000 trajectories: 100,000 positive and 100,000 negative. MAE_all and RMSE_all cover all truly positive trajectories, including detection false negatives; MAE_tp and RMSE_tp cover only true-positive detections. Localization errors use trajectory-position units. These metrics are not interchangeable and are not directly compared to papers using different definitions.

## Independent-test summary

Arithmetic mean ± **sample** standard deviation (`ddof=1`) across the five seeds; six-decimal display only. Machine-readable exports retain the full numeric precision read from the JSONs. All models share the same test split, so the SD describes across-seed/model variability, not independent test-sample uncertainty or a confidence interval.

| Architecture | f1 | fpr | mae_all |
|---|---:|---:|---:|
| LSTM | 0.796939 ± 0.001376 | 0.191380 ± 0.003851 | 9.260193 ± 0.092847 |
| xLSTM | 0.791724 ± 0.002578 | 0.184418 ± 0.007154 | 9.670149 ± 0.146424 |
| CNN-LSTM | 0.792946 ± 0.001461 | 0.180558 ± 0.002992 | 7.593484 ± 0.041192 |
| Transformer | 0.797062 ± 0.006503 | 0.188006 ± 0.004397 | 7.675603 ± 0.177171 |
| ConvTransformer | 0.805349 ± 0.002929 | 0.191190 ± 0.003967 | 7.286198 ± 0.211986 |

| Architecture | mae_tp | rmse_all | rmse_tp |
|---|---:|---:|---:|
| LSTM | 7.717776 ± 0.174435 | 13.300407 ± 0.051372 | 11.455639 ± 0.115655 |
| xLSTM | 8.085820 ± 0.124900 | 13.525134 ± 0.080561 | 11.651995 ± 0.077817 |
| CNN-LSTM | 5.950245 ± 0.084170 | 11.592371 ± 0.041682 | 9.784101 ± 0.089307 |
| Transformer | 6.033789 ± 0.103678 | 11.660367 ± 0.150174 | 9.870992 ± 0.096304 |
| ConvTransformer | 5.737142 ± 0.211945 | 11.432403 ± 0.113609 | 9.696333 ± 0.119221 |

| Architecture | accuracy | precision | recall |
|---|---:|---:|---:|
| LSTM | 0.798913 ± 0.001091 | 0.804845 ± 0.002538 | 0.789206 ± 0.003964 |
| xLSTM | 0.795849 ± 0.001120 | 0.808059 ± 0.004246 | 0.776116 ± 0.008694 |
| CNN-LSTM | 0.797494 ± 0.000958 | 0.811162 ± 0.001906 | 0.775546 ± 0.003902 |
| Transformer | 0.799602 ± 0.005460 | 0.807214 ± 0.003957 | 0.787210 ± 0.010701 |
| ConvTransformer | 0.805918 ± 0.002531 | 0.807708 ± 0.002875 | 0.803026 ± 0.005524 |

ConvTransformer achieved the highest observed mean F1 on the independent test under the experimental protocol considered. This descriptive observation does not establish universal superiority or statistical significance; no inferential statistical test is claimed.

## Validation-calibration versus test

[validation_vs_test.csv](validation_vs_test.csv) keeps both splits in separate named columns for all nine metrics, with mean, sample SD and the descriptive difference `test minus calibration`. The same five seeds are paired within each architecture. Calibration metrics were measured on data used to choose thresholds, whereas test metrics use the frozen thresholds on the held-out split. Differences are descriptive, not a new selection phase, threshold optimization, or significance test. An observed test FPR is not constrained to be ≤ 0.20 by recalibration; the cap governed the earlier calibration selection only.

## Files and provenance

- [independent_test_25_runs.csv](independent_test_25_runs.csv): all 25 identities, statuses, exact metrics, confusion counts and frozen thresholds.
- [architecture_summary.csv](architecture_summary.csv): nine metrics with mean, sample SD, minimum and maximum per architecture.
- [validation_vs_test.csv](validation_vs_test.csv): separate split summaries, 45 architecture/metric rows.
- [source_provenance.json](source_provenance.json): archive hash, date, identities, per-JSON hashes, seal hashes, complete source metrics and sanitized authorization metadata. Identities come from the archived final plan and JSON filenames; the original JSONs themselves do not contain run_id.

Archive SHA-256: `f770b3543cdb0c8db8f1c5fd07e18a6454f21fdf1f6262b3e9169b9c71e47b88`.

The archive's per-result checksum manifest and per-run analysis CSV were cross-checked against the 25 source JSONs. Architecture statistics here are recomputed from those JSON metrics. No HDF5, checkpoint, archive, raw log, private absolute path or credential is distributed. The source hashes enable comparison with the archive; the public derived proof is not a substitute for the original experimental package.

## Read-only verification

```bash
python -B scripts/summarize_independent_test_results.py
```

Run from the repository root. The standard-library validator checks identities, source-metric agreement, pre-test seal/threshold consistency, nine-metric summaries, validation/test comparisons and README tables without opening datasets or checkpoints or writing files.
