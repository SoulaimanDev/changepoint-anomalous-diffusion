# Final TFM additions

This folder freezes the two additional experiments retained for the final TFM:

1. `ConvTransformer-v2` multi-seed robustness study at `L=100`.
2. Classical PELT baseline evaluated on the full validation and test partitions.

## ConvTransformer-v2 multi-seed protocol

- Seeds: `1, 2, 3, 42, 123`.
- Fixed balanced training subset: 10,000 trajectories.
- Subset-selection seed: `2026`.
- Full validation partition: 20,000 trajectories.
- Full test partition: 200,000 trajectories.
- Epoch limit: 20.
- Batch size: 1024.
- Learning rate: `1e-3`.
- Detection class weights: positive `1.0`, negative `1.8`.
- Multitask loss weights: `[2.0, 1.0]`.
- Threshold selected independently on validation for each seed.

This is a reduced-training robustness experiment. It must not replace the
single main v2 execution trained with the complete original protocol.

Aggregated test results:

| Metric | Mean +/- std |
|---|---:|
| Accuracy | 0.5627 +/- 0.0869 |
| Recall | 0.8227 +/- 0.2453 |
| F1 | 0.6469 +/- 0.0363 |
| FPR | 0.6974 +/- 0.4144 |
| MAE all | 14.4826 +/- 1.1010 |
| RMSE all | 17.0397 +/- 0.8186 |

Seeds `2`, `3`, and `123` converge to an almost-always-positive detector.
This reduced protocol should not be compared directly with the main full-training
execution as if both experiments shared the same training conditions.

## PELT protocol

- Features: standardized instantaneous energy `dx(t)^2` and absolute increment
  `|dx(t)|`.
- Cost: `l2`.
- Minimum segment size: 10.
- Jump: 5.
- Penalty: 8.0.
- Full validation partition: 20,000 trajectories.
- Full test partition: 200,000 trajectories.

Penalty 8.0 was chosen in a preliminary validation-only grid and was then
applied to the complete validation and test partitions. Test data were not used
to choose the penalty.

Final test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.534055 |
| Precision | 0.534618 |
| Recall | 0.525930 |
| F1 | 0.530238 |
| FPR | 0.457820 |
| FNR | 0.474070 |
| MAE TP | 16.316563 |
| RMSE TP | 22.334756 |
| MAE global with FN penalty | 55.988369 |
| RMSE global with FN penalty | 70.732285 |

For PELT, an undetected real changepoint receives a localization penalty of 100
time points in the global MAE/RMSE. Therefore, global localization errors are
not directly equivalent to the neural `all` metric. The true-positive MAE/RMSE
provide the cleaner conditional localization comparison.
