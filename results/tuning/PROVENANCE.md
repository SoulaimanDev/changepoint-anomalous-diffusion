# Provenance of the complete tuning export

Source archive: `tfm_campaign_v2_BACKUP_2026-09-14.tar.gz`.
SHA-256: `8ece376a4122cab455c18b864ac83c0438c4c6b49a5c1bcb1091ac3c2b7d8db8` (verified before extraction).

The original archive contains 450 tuning run.json records, ten execution registries and their checkpoints. It does not contain tuning_merged.csv or selected_configs.json. Both were reconstructed locally using the unchanged archived merger and selector, including code inventory, result hashes, checkpoint hashes, identities and frozen hyperparameter checks. No values were supplied to selection from the independently provided expected results. All 20 independent comparisons matched exactly.

Only tuning results are included; no independent final/test results, datasets, checkpoints, credentials, billing records or active cloud state are published.

## Verified digests

| Item | SHA-256 |
|---|---|
| Code manifest | `a91e1ddc8c99c8398f679ed10f5f8b057b433f622378ab85d8f5cbaa136094c9` |
| Protocol (archived protocol_sha256.txt) | `90c8471c8360659deafba8c53bc2d3ff4d2f025e244807ae66997cb79a46a367` |
| Canonical plan | `0d7a1171cb9cd7e994ce86df96907e3718a200a439ed6f9c167bd3a3e352fc85` |
| Train (recorded input hash) | `366f94e0962a164bcf63d235ea26540777fae7958014de33c54de7f98c9e5d35` |
| Validation (recorded input hash) | `2d7a680dd9cc17a3e64d5a1f20621e7d99332c375245dc5a1eaabdf53532c51d` |
| Validation split (recorded input hash) | `2cce9795633852ff228a625cbee3e587461a231e358d1485fc0c44f76da0718a` |
| Runtime requirements lock | `a8ef580807ac716c39679f3aa68f804123cb5d9a672eeda64478c2370337b97e` |
| Packages (recorded environment fingerprint) | `6ed00d7c5a26f76e99227ab72eea7af76ec7fef4c379d28ffa614839103acc2f` |

Data hashes are verified against archived frozen_inputs.json; datasets were not opened or rehashed during this publication task. Environment values come from reports/environment_report.json and the archived smoke receipt, not a fresh GPU execution.

## Archived training environment

- gpu: `NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb`
- python: `3.11.9`
- torch: `2.11.0+cu128`
- cuda: `12.8`
- cudnn: `91900`
- numpy: `1.26.4`
- h5py: `3.14.0`
- pyyaml: `6.0.2`

## Exact source paths inside the archive

- `tfm_campaign_v2/code/runpod/execution.py`: merged_rows, merge and verify_merge.
- `tfm_campaign_v2/code/runpod/core.py`: terminal_record, code inventory and checkpoint/hash checks.
- `tfm_campaign_v2/code/runpod/final_phase.py`: choices_from_merge and select_configs.
- `tfm_campaign_v2/code/confirmatory/selection.py`: threshold, checkpoint and configuration ordering; arithmetic means use NumPy.
- `tfm_campaign_v2/code/confirmatory/frozen.py`: frozen design and hyperparameter checks.
- `tfm_campaign_v2/code/configs/confirmatory_step6_L100_selection.yaml`: frozen selection contract.
- `tfm_campaign_v2/code/runpod/manifests/frozen_inputs.json`: train, validation and split digests.
- `tfm_campaign_v2/reports/environment_report.json` and `reports/smoke_test/20260910_225531/runtime_requirements.lock`: environment evidence.
- `tfm_campaign_v2/results/shard_00` through `shard_09`: original execution registries and tuning records.

Source code hashes are recorded below so these paths can be checked in the original backup. The backup itself is not included in Git.

| Archived source relative to code/ | SHA-256 |
|---|---|
| `runpod/execution.py` | `22eeeb01841cbb400a6ac4b951a2489b70aa6fae47708028770ff98a3ea4e5a0` |
| `runpod/core.py` | `6b888009a90143bb7b11adaafe72dfe9c92b538b150de9b6bb37b5f02ca8806a` |
| `runpod/final_phase.py` | `45f5cf75fa66c3cfd0a05f5da925cbc11223198c080d3cd1716a4233db46a92c` |
| `confirmatory/selection.py` | `4edfe04496e96aff82f8db87215c16fc75a445229a4041bf275776b1799de5f8` |
| `confirmatory/frozen.py` | `d4e53fe332061b7c1503d84ea405c76c7046e44f6035e1c284703614bd0a04d4` |
| `configs/confirmatory_step6_L100_selection.yaml` | `a32c66626bd224292f79a4a9ba2a28a4c6241d77924d5f928be0c3d0b91c59ad` |

Local reconstruction used Python 3.11.0 and NumPy 1.26.4. This was CPU-side metadata aggregation and hash verification, not training. Original inputs and archived scientific code remained unchanged.
