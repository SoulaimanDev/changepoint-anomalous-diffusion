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
- Threshold selected independently on validation for each seed.

This is a reduced-training robustness experiment. It must not replace the
single main v2 execution trained with the complete original protocol.

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

For PELT, an undetected real changepoint receives a localization penalty of 100
time points in the global MAE/RMSE. Therefore, global localization errors are
not directly equivalent to the neural `all` metric. The true-positive MAE/RMSE
provide the cleaner conditional localization comparison.
