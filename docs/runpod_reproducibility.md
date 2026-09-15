# RunPod reproducibility record

> **Historical status — September 2026 preparation record.** References below to "this branch" or "this documentation commit" describe the earlier documentation-only publication, not all files now in `main`. The later offline reconstruction and completed 450-run checks are documented in [current tuning provenance](../results/tuning/PROVENANCE.md). The operational record is preserved; the independent final phase is not included in this update.


This document describes operational safeguards reported by the campaign owner for `tfm_campaign_v2`. It does not provide instructions to operate the active Pods. Scientific scope, supplied hashes and environment versions are recorded in [the campaign document](confirmatory_campaign_2026.md).

## Preparation and validation

Bootstrap belongs to the validated offline tooling and prepares the locked environment before execution. Its exact implementation has not been integrated in this documentation commit. Reconstructing a bootstrap from historical requirements would not establish equivalence with the validated package.

The owner reported an official preflight of 102 passing tests with no failures or skips, followed by successful GPU smoke validation for LSTM, xLSTM, CNN-LSTM, Transformer and ConvTransformer. A smoke test checks construction, data flow, forward/backward and environment compatibility; it does not estimate final accuracy or establish superiority. Neither bootstrap nor smoke validation was executed during this update.

## Identity, shards and integrity

There are ten shards, `shard_00` through `shard_09`, with 45 scientific run identities per shard. Architecture, configuration and seed identify the scientific run independently of its physical execution location. A later offline audit should confirm disjoint shard membership, no duplicates or missing identities, and agreement with the canonical 450-run plan. This document does not report a new audit of live registries.

The per-shard `.writer.lock` rejects concurrent writers with `Shard already locked; no concurrent writer permitted`. This prevents conflicting writes and duplicate work within that shard. A lock error alone is not evidence that a lock is stale. This update neither removes locks nor changes concurrency behavior.

## Checkpoints and continuation

The reported restart behavior retained completed runs and continued remaining planned runs. Existing checkpoints, run records and attempt provenance must remain associated with their original identities. Preserving completed work is distinct from promising that an interrupted training loop resumes at an exact minibatch: that depends on the validated runner and checkpoint contract.

No new policy for interrupted `running` records, numerical failures, attempt retries or checkpoint restoration is introduced here. Do not silently replace a failed run with a different seed or configuration. The original rules remain authoritative; an offline source/record audit is necessary before describing additional recovery guarantees.

## Billing attestation guard

A freshness guard can prevent new work when a billing attestation is stale or future-dated. The observed error was `ValueError: Stale/future attestation; refresh from console`. This is an operational guard, not a model-selection rule. Its consequences for already-running work are not inferred here. No billing file was read or refreshed during this update.

Do not commit balances, card information, API keys, tokens, S3 credentials, private resource identifiers or financial records. The ignore rules cover common private files and runtime outputs, but ignore rules do not remove already tracked files or replace review of staged content. Any future public provenance export must be reviewed for sensitive fields before publication.

## Repository and active infrastructure are separate

This branch modifies Markdown documentation and `.gitignore` only. It does not synchronize GitHub to a Pod, install dependencies, start training, modify a shared volume or alter a scientific file. Local unit tests exercise repository helpers independently of the official GPU campaign. No remote GPU test is required for this documentation change.

Later source integration must use an approved offline archive matching the supplied code digest, preserving scientific source bytes and validating its hashing procedure. A Git commit hash is not the same object as the campaign code SHA-256. Keep both provenance identifiers rather than treating one as a replacement for the other.
