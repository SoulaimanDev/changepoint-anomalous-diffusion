# Confirmatory multi-architecture campaign — September 2026

> **Historical status — 11 September 2026; superseded for tuning on 14 September 2026.** The text below preserves the then-current campaign, authorization and source-availability record. The exact offline archive was subsequently identified and verified, and all 450 tuning runs were accounted for; see [completed tuning and reconstruction](../results/tuning/README.md) and [provenance](../results/tuning/PROVENANCE.md). Statements below about pending audits, missing source packages or publication plans describe that earlier stage. The independent final phase is separate and is not included in this update.


## Scope and status of this record

Campaign: `tfm_campaign_v2`. Status: **in progress**, as reported by the campaign owner for this documentation update on 11 September 2026. This is a supplied validation record, not a live dashboard. No active Pod or campaign file was accessed to prepare it. The reported preflight and GPU checks were not repeated locally.

The scientific objective is binary changepoint detection and temporal localization in one-dimensional anomalous-diffusion trajectories. This confirmatory comparison is separate from the historical main L=100 execution, reduced-training ConvTransformer-v2 multiseed study, PELT baseline, L=200 extension and exploratory ConvTransformer-v3 work. Their existing tables remain historical evidence under their own protocols. The historical `frozen_v2` name does not identify the new campaign's code or environment.

## Frozen experimental plan

| Item | Confirmatory plan |
|---|---|
| Architectures | LSTM, xLSTM, CNN-LSTM, Transformer, ConvTransformer |
| Configurations | 30 per architecture |
| Tuning seeds | 11, 29, 47 |
| Tuning identities | 5 × 30 × 3 = 450 |
| Shards | shard_00 through shard_09; 45 identities each |
| Planned final seeds | 1, 2, 3, 42, 123 |
| Planned final runs | 5 × 5 = 25; not yet authorized |

CNN-LSTM denotes the Conv1D → LSTM architecture, not a ConvLSTM recurrent cell. ConvTransformer denotes the campaign implementation; this record does not substitute an exploratory v3 model for it.

Training uses train. Hyperparameter and model selection use validation only, with tuning and calibration roles kept separate under the frozen split and protocol. Calibration is not a source of tuning feedback. Test is excluded from tuning, ranking, feasibility decisions, checkpoint selection and threshold selection. It remains reserved for final evaluation after architecture, hyperparameters, checkpoint and threshold are frozen. This record does not authorize that evaluation.

The frozen protocol remains authoritative for normalization, losses, metrics, masks, thresholds, feasibility, checkpoint and ranking rules, determinism and hyperparameter spaces. None of those rules is redefined here. The future final phase requires explicit approval; no final runs or synthetic final results are created by this update.

## Validated provenance supplied by the campaign owner

| Artifact | SHA-256 |
|---|---|
| Campaign code | `a91e1ddc8c99c8398f679ed10f5f8b057b433f622378ab85d8f5cbaa136094c9` |
| Protocol | `90c8471c8360659deafba8c53bc2d3ff4d2f025e244807ae66997cb79a46a367` |
| Canonical run plan used for the 10-shard campaign | `0d7a1171cb9cd7e994ce86df96907e3718a200a439ed6f9c167bd3a3e352fc85` |
| Train | `366f94e0962a164bcf63d235ea26540777fae7958014de33c54de7f98c9e5d35` |
| Validation | `2d7a680dd9cc17a3e64d5a1f20621e7d99332c375245dc5a1eaabdf53532c51d` |
| Validation split | `2cce9795633852ff228a625cbee3e587461a231e358d1485fc0c44f76da0718a` |

The canonical plan hash identifies the supplied plan artifact; it is not a claimed hash of each shard manifest or their concatenation. A later offline provenance audit must check the exact manifests against that plan.

The reported official preflight was `READY_FOR_GPU`: **102/102 tests passed, 0 failed, 0 skipped**, with `official_training_runs = 0` at preflight time. The latter is not the current completion count. The GPU smoke test passed for all five architectures. These checks concern technical validity, not comparative scientific performance. Local repository unit tests are a separate suite and cannot re-certify this GPU preflight.

| Environment component | Validated value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb |
| Python | 3.11.9 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| NumPy | 1.26.4 |
| h5py | 3.14.0 |
| PyYAML | 6.0.2 |

The smoke-test dependency package was reported locked and validated. The historical repository's broad dependency ranges are not a replacement for that lock. This documentation does not install or alter the active environment.

## Sharding, concurrency and continuation

The plan distributes architecture identities deterministically across ten shards, each containing 45 identities, for a total of 450. Sharding organizes execution without changing scientific identities or selection rules.

A per-shard `.writer.lock` prevents concurrent writers. A duplicate launch was reported correctly refused with `Shard already locked; no concurrent writer permitted`. This is an integrity protection, not a scientific failure. Its operation is not changed here.

The reported continuation preserved `completed` runs and continued the remaining `planned` runs. Completed identities must not be retrained as part of ordinary continuation. This is an operational property, not a new experimental protocol. Handling an interrupted nonterminal attempt requires the exact runner's existing rules; this document does not authorize rewriting statuses, removing locks, changing seeds or selecting a favorable retry. See [the infrastructure record](runpod_reproducibility.md).

## Results and publication boundary

An intermediate owner observation reported CNN-LSTM tuning at 90/90 completed. This is not a current global count, a final model selection or evidence of superiority. Partial tuning results are for technical monitoring only. No winner or final performance table is asserted here.

After the 450 identities are accounted for, selection must follow the existing frozen rules. The planned 25-run final phase remains subject to explicit approval. Only after the authorized final procedure is complete should definitive results be published in [results/confirmatory_2026](../results/confirmatory_2026/README.md), with public provenance and without private runtime artifacts.

## Offline source integration remains deferred

No exact offline source package matching the validated campaign code hash was identified in the supplied local materials. A differently hashed local preparation is not substituted for it. This commit therefore contains documentation and Git exclusions only; it does not claim to distribute the validated campaign implementation.

A later, separately reviewed integration needs the exact validated offline archive: the complete scientific `confirmatory/` source tree, complete `runpod/` orchestration tree including its bootstrap and runner, frozen configuration/protocol files, canonical plan and all ten shard manifests, validation split manifest and identity metadata, code file-hash manifest and hashing procedure, dependency lock, and original preflight/smoke reports. Preserve source bytes and original relative paths as recorded by that archive; filenames absent from the archive are not to be invented. Large data, checkpoints, live registries and private billing material do not belong in Git. Do not obtain these files by contacting an active Pod for this documentation task.
