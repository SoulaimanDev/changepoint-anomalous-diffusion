# Frozen v2 results

This folder freezes the results of the current project version before introducing ConvTransformer-v3, multi-seed repetitions, or a PELT baseline.

The purpose is reproducibility and methodological hygiene: the current v2 metrics are preserved as the reference baseline, and future experiments should write to new folders instead of overwriting these files.

## Scope

- Main length for the four-architecture comparison: `L=100`.
- Architectures: LSTM, CNN-LSTM, Transformer, ConvTransformer-v2.
- Main task: binary changepoint detection plus temporal localization.
- Input used by v2 notebooks: normalized increments `dx(t)`.
- Reported v2 status: one main execution per architecture, not a multi-seed study.

## Files

- `global_detection_metrics_L100.csv`: threshold, accuracy, precision, recall, F1, FPR and FNR for the four architectures.
- `localization_metrics_L100.csv`: MAE/RMSE for all positive trajectories and for true positives.
- `fpr_no_changepoint_by_generator_L100.csv`: false positive rate on homogeneous trajectories by diffusion generator.
- `convtransformer_length_summary_v2.csv`: descriptive ConvTransformer-v2 comparison between `L=100` and `L=200`.
- `run_manifest.json`: metadata, source files and limitations of the freeze.

## Important limitation

The `L=100` versus `L=200` ConvTransformer-v2 comparison is stored for traceability, but it should not be interpreted as a strictly controlled ablation of trajectory length. The training batch size, loss weighting and threshold grid differ between the two runs documented in the thesis.

