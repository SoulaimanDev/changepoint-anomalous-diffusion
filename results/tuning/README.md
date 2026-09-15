# Complete tuning campaign — 450 runs

The archived campaign completed **450/450 tuning runs**, with **0 failed**: five architectures, 30 configurations each, seeds **11, 29, 47**, 90 runs per architecture, and 10 shards of 45 runs. This is the complete tuning state in the backup dated **14 September 2026**.

Training used train and validation_tuning only. No final test data or final test results are included. These validation/tuning results select one configuration per architecture; they are not the independent final evaluation. The separate 25-run final phase and subsequent test evaluation are outside this export.

## Selected configurations

| Architecture | Selected config | Mean validation F1 | Mean MAE_all | Mean FPR | Valid seeds |
|---|---|---:|---:|---:|---:|
| LSTM | cfg_02 | 0.8006514266130286 | 9.60602887471517 | 0.19066666666666668 | 3 |
| xLSTM | cfg_09 | 0.7940252554167891 | 9.782986640930176 | 0.19166666666666665 | 3 |
| CNN-LSTM | cfg_06 | 0.800410476840066 | 7.6136603355407715 | 0.19226666666666667 | 3 |
| Transformer | cfg_06 | 0.8022087901770522 | 7.887541135152181 | 0.18613333333333335 | 3 |
| ConvTransformer | cfg_12 | 0.8097509375627459 | 7.431063175201416 | 0.19766666666666666 | 3 |

ConvTransformer achieved the highest mean validation F1 among the evaluated configurations under the frozen tuning protocol. Transformer has the second-highest mean F1 among the selected configurations. ConvTransformer and CNN-LSTM have the lowest mean MAE_all among these selections. xLSTM does not outperform LSTM on mean validation F1 in this tuning campaign. These differences do not establish statistical significance or definitive generalization performance; no new statistical test is claimed.

## Exact frozen selection

- Threshold: evaluate 101 thresholds from 0 to 1 in steps of 0.01; require FPR <= 0.20 (implementation tolerance 1e-15); maximize F1, minimize FPR, then prefer the highest threshold.
- Checkpoint/epoch: maximize F1, minimize MAE_all, then minimize FPR. The same improvement event governs checkpoint saving, early stopping and LR patience.
- Configuration: require all three seeds 11, 29, 47 to be completed, finite and selection-feasible; take their arithmetic mean; maximize mean F1, minimize mean MAE_all, then minimize mean FPR. No seed is omitted. Complete ties retain the first candidate in the archived design order.

The archived selector trusts each run's selection_feasible flag; the public validation script additionally checks FPR feasibility directly. MAE_all includes every truly positive validation trajectory, including detection false negatives.

## Reconstruction and files

The original 450 runs are present in the verified backup. **tuning_merged.csv and selected_configs.json were absent from the original archive**. They were reconstructed locally from those runs by direct calls to the exact archived `runpod.execution.merge` and `runpod.final_phase.select_configs`. No scientific code, rule or original metric was changed. The two reconstructed files are copied here byte-for-byte.

- [tuning_merged.csv](tuning_merged.csv): 450 registry rows, including original run hashes and source shards; it is not a per-run metric table.
- [selected_configs.json](selected_configs.json): official selector output, with all 150 candidate summaries and the five selections.
- [tuning_summary.csv](tuning_summary.csv): automatic export of the five selected summaries.
- [tuning_run_metrics.csv](tuning_run_metrics.csv): compact metric-only export from the 450 archived run.json files for portable validation.
- [scientific_comparison.json](scientific_comparison.json): exact independent comparison with the supplied expected selections and metrics, 20/20 PASS.
- [reconstruction_record.json](reconstruction_record.json) and [tuning_merged.manifest.json](tuning_merged.manifest.json): reconstruction record and official merge receipt. Absolute local paths are historical provenance, not portable execution paths.
- [PROVENANCE.md](PROVENANCE.md): archive, code, protocol, data and environment digests.

Run from the repository root:

```bash
python -B scripts/summarize_tuning_results.py
```

The script is read-only, needs only the Python standard library, validates all 450 identities and all 150 candidate summaries, checks the five selections and prints the summary. It does not train, load checkpoints, or access datasets. It does not replace the original archived merger's checkpoint/hash checks, which already passed during reconstruction.

The older [interim directory](../confirmatory_2026/README.md) is historical. Its snapshots must not be mistaken for the completed tuning state.

## Byte-level hashes and Windows checkouts

Recorded hashes identify the original/archived artifact bytes, including the reconstructed exports as published. Git's automatic line-ending conversion on Windows can change checkout bytes (LF to CRLF), even when the JSON content is equivalent, and consequently cause byte-level validation to fail. A mismatch must be investigated against the original Git blob/archive; do not regenerate the reference hashes or edit result values to make the check pass. Line-ending policy is deferred to a separate technical review; this documentation update changes neither artifacts nor hashes.
