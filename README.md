# Changepoint Detection in Anomalous Diffusion

Reproducible code and frozen artifacts for the September 2026 Master's thesis on changepoint detection and temporal localization in synthetic anomalous-diffusion trajectories.

The scientific objective is deliberately narrow: given a one-dimensional trajectory generated from anomalous-diffusion processes, decide whether it contains one model transition and, when a transition is present, estimate its temporal position. The repository does not claim to identify the diffusion model or infer the anomalous exponent inside each segment.

## Confirmatory multi-architecture campaign (September 2026)

A separate confirmatory multi-seed campaign is in progress, according to the campaign owner's September 2026 record: LSTM, xLSTM, CNN-LSTM, Transformer and ConvTransformer, with 30 configurations per architecture and tuning seeds `11, 29, 47` (450 tuning runs). Selection uses validation only; test is reserved for final evaluation after all selection decisions are frozen. Execution is distributed over 10 shards, with preflight, GPU smoke validation, provenance checks and concurrent-writer locks.

Definitive results will be published only after completion of the campaign and the explicitly approved final phase. See [the campaign record](docs/confirmatory_campaign_2026.md) and [infrastructure reproducibility](docs/runpod_reproducibility.md). This documentation update does not deploy code or operate the active campaign.

## Scientific provenance

This project is built around the AnDi ecosystem, but it uses a thesis-specific supervised protocol rather than the original challenge tasks.

- Muñoz-Gil et al. (2021), DOI `10.1038/s41467-021-26320-w`, provides the AnDi context and the distinction between T1 anomalous-exponent inference, T2 diffusion-model classification, and T3 trajectory segmentation.
- Bo et al. (2019), DOI `10.1103/PhysRevE.100.010102`, and Argun et al. (2021), DOI `10.1088/1751-8121/ac070a`, provide recurrent-neural-network precedents for anomalous-diffusion characterization, including the use of recurrent models on short single trajectories.
- Garibo-i-Orts et al. (2021), DOI `10.1088/1751-8121/ac3707`, is a domain-specific precedent for convolution + BiLSTM sequence modelling.
- Vaswani et al. (2017), "Attention Is All You Need", is the general Transformer reference.
- Firbas et al. (2023), DOI `10.1088/1751-8121/acafb3`, is a domain-specific precedent for convolutional Transformer models in anomalous-diffusion characterization.

The thesis CNN-LSTM and ConvTransformer models are adaptations for binary changepoint detection and localization. They are not exact reproductions of the Garibo-i-Orts or Firbas architectures.

## Historical experiments

The data descriptions, architecture details, result tables and exploratory studies below describe the historical experiments. Their numbers and reproduction commands are preserved; they are not results or launch instructions for the new 450-run confirmatory campaign.

## Data and protocols

Synthetic trajectories are generated with the five theoretical AnDi models used throughout the thesis:

- Annealed Transient Time Motion (`ATTM`)
- Continuous Time Random Walk (`CTRW`)
- Fractional Brownian Motion (`FBM`)
- Levy walk (`LW`)
- Scaled Brownian Motion (`SBM`)

| Protocol | Length | Task | Train | Validation | Test |
|---|---:|---|---:|---:|---:|
| A | 100 | Localization with a guaranteed changepoint | 200,000 | 20,000 | 200,000 |
| B | 100 | Binary detection plus localization | 200,000 | 20,000 | 200,000 |
| B | 200 | ConvTransformer extension | 200,000 | 20,000 | 200,000 |

For Protocol B, each split is balanced between trajectories with one changepoint and trajectories without a changepoint. Positive samples are balanced across the 20 ordered transitions between distinct models; negative samples are balanced across the five single-model generators.

The changepoint is restricted to the central 20-80 percent of the trajectory:

- `L=100`: `cp` in `[20, 80]`, increment index `cp_dx = cp - 1` in `[19, 79]`
- `L=200`: `cp` in `[40, 160]`, increment index `cp_dx = cp - 1` in `[39, 159]`

The binary neural models use per-trajectory standardized increments `dx = diff(X)` as input. This is an offline normalization: the mean and standard deviation are computed from the full observed sequence.

## Architectures

- `LSTM`: recurrent baseline with two temporal layers.
- `CNN-LSTM`: one-dimensional convolutional feature extraction followed by LSTM layers.
- `Transformer`: learned positional encoding and Transformer encoder blocks.
- `ConvTransformer-v2`: local convolutional stem and residual convolutional blocks followed by Transformer encoder blocks.

The historical notebook filenames `03_convlstm_...` and `09_convlstm_...` are kept for reproducibility. Those files implement a Conv1D -> LSTM pipeline, not a ConvLSTM recurrent cell.

All Protocol B neural networks have one binary detection output and one localization output over valid changepoint positions. Localization loss is masked for negative samples. Reported point estimates use the expectation of the predicted localization distribution.

## Main L=100 results

The following values come from the frozen single main execution stored in `results/frozen_v2/`. Localization errors labelled `all` include every truly positive trajectory, including false negatives.

| Model | Threshold | Accuracy | Precision | Recall | F1 | FPR | FNR | MAE all | RMSE all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.35 | 0.7328 | 0.7341 | 0.7300 | 0.7320 | 0.2645 | 0.2700 | 12.7331 | 16.3504 |
| CNN-LSTM | 0.25 | 0.7891 | 0.7980 | 0.7743 | 0.7860 | 0.1960 | 0.2257 | 7.6645 | 11.9183 |
| Transformer | 0.30 | 0.7712 | 0.8093 | 0.7095 | 0.7561 | 0.1672 | 0.2905 | 8.7414 | 12.5307 |
| ConvTransformer-v2 | 0.30 | 0.8037 | 0.8184 | 0.7806 | 0.7990 | 0.1732 | 0.2194 | 7.5370 | 11.5706 |

Transformer gives the lowest false-positive rate in this run, while ConvTransformer-v2 gives the best observed compromise between detection and localization. These are single-run results, not multi-seed estimates.

## ConvTransformer-v2 multiseed study

The multiseed experiment that is actually available in this repository covers ConvTransformer-v2 only:

- Seeds: `1, 2, 3, 42, 123`
- Balanced training subset: 10,000 trajectories
- Validation: 20,000 trajectories
- Test: 200,000 trajectories
- Epochs: 20
- Batch size: 1024
- Learning rate: `1e-3`
- Detection class weights: positive `1.0`, negative `1.8`
- Multitask loss weights: `[2.0, 1.0]`

| Metric | Mean +/- std |
|---|---:|
| Accuracy | 0.5627 +/- 0.0869 |
| Recall | 0.8227 +/- 0.2453 |
| F1 | 0.6469 +/- 0.0363 |
| FPR | 0.6974 +/- 0.4144 |
| MAE all | 14.4826 +/- 1.1010 |
| RMSE all | 17.0397 +/- 0.8186 |

Three seeds (`2`, `3`, and `123`) converge to an almost-always-positive detector. Because this is a reduced-training robustness experiment, it should not be compared directly with the full-training main execution as if the protocols were identical.

## PELT baseline

`scripts/run_pelt_baseline_L100.py` implements an elementary classical baseline with `ruptures`.

The retained configuration uses:

- Cost: `l2`
- Features: standardized `dx(t)^2` and `|dx(t)|`
- Minimum segment size: `10`
- PELT jump: `5`
- Local contrast window: `5`
- Missing-localization penalty for false negatives: `100`
- Penalty selection: validation only
- Final selected penalty: `8.0`

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

PELT is included as a simple classical reference point. These results do not prove a general superiority of deep learning over classical methods.

## ConvTransformer-v2 at L=200

Notebook `13_convtransformer_L200_detection_localization.ipynb` and `results/L200/` are preserved from `main` as the final GLOBAL `L=200` execution.

| Threshold | Accuracy | Precision | Recall | F1 | FPR | FNR | MAE | RMSE | nMAE | nRMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.475 | 0.852365 | 0.894449 | 0.799020 | 0.844046 | 0.094290 | 0.200980 | 9.376101 | 18.274097 | 0.0469 | 0.0914 |

The executed notebook uses positive detection class weight `1.5`, negative detection class weight `1.0`, and multitask loss weights `[2.0, 1.0]`. These parameters are part of the saved run and should not be rewritten as `1.0/1.0`.

The `L=100` versus `L=200` comparison is descriptive: length, threshold grid, batch size, and class weighting are not held under a single controlled ablation.

## ConvTransformer-v3 exploratory work

ConvTransformer-v3a and ConvTransformer-v3b remain exploratory. No test metrics are reported for either variant, and they are not presented as better than ConvTransformer-v2.

v3a replaces the single-channel increment input with six channels:

1. standardized increment
2. absolute value
3. square
4. local variance with window 5
5. local mean of absolute value with window 5
6. lag-1 product

These features are computed only from the observed signal. They do not use the binary label, the true changepoint, or the generator-model identity.

v3b keeps the same multichannel input and changes the localization target to a discrete Gaussian distribution centered on the true changepoint. The documented exploratory trials concern `sigma=1` and `sigma=2`.

## Repository structure

```text
.
|-- configs/
|   |-- convtransformer_v3_L100.yaml
|   |-- frozen_v2.yaml
|   `-- seeds.yaml
|-- data_synthetic_changepoint_andi/
|-- docs/
|   |-- experiment_protocol_v2.md
|   `-- experiment_protocol_v3.md
|-- notebooks/
|   |-- 01_synthetic_dataset_changepoint_anomalous_diffusion.ipynb
|   |-- 02_lstm_changepoint_detection.ipynb
|   |-- 03_convlstm_changepoint_detection.ipynb
|   |-- 04_transformer_changepoint_detection.ipynb
|   |-- 05_convtransformer_changepoint_detection.ipynb
|   |-- 06_model_validation_changepoint_comparison.ipynb
|   |-- 07_synthetic_dataset_binary_changepoint.ipynb
|   |-- 08_lstm_binary_detection_localization.ipynb
|   |-- 09_convlstm_binary_detection_localization.ipynb
|   |-- 10_transformer_binary_detection_localization.ipynb
|   |-- 11_convtransformer_binary_detection_localization.ipynb
|   |-- 12_synthetic_dataset_binary_changepoint_L200.ipynb
|   `-- 13_convtransformer_L200_detection_localization.ipynb
|-- results/
|   |-- frozen_v2/
|   |-- final_tfm_additions/
|   |-- L200/
|   |-- multiseed_v2_L100_reduced10k_convtransformer_only/
|   `-- pelt_baseline_L100_full_penalty8/
|-- scripts/
|   |-- aggregate_multiseed_v2_results.py
|   |-- build_synthetic_changepoint_dataset.py
|   |-- build_synthetic_with_without_changepoint_dx.py
|   |-- feature_engineering.py
|   |-- label_engineering.py
|   |-- run_convtransformer_v3a_L100.py
|   |-- run_convtransformer_v3b_L100.py
|   |-- run_multiseed_v2_L100.py
|   |-- run_pelt_baseline_L100.py
|   `-- run_v2_single_seed_L100.py
|-- tests/
`-- requirements.txt
```

`docs/thesis/` is intentionally absent from this synchronized branch. The final September 2026 manuscript sources are kept separate until the final LaTeX source tree is explicitly provided.

Generated datasets (`*.h5`), trained models, and intermediate training outputs are intentionally excluded from Git. The tracked CSV, JSON, YAML, Markdown, and PNG artifacts are the source of truth for the results shown here.

## Reproduction commands

Create an isolated environment and install the declared dependency ranges:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run lightweight tests:

```bash
python -m unittest discover -s tests -v
```

Generate Protocol A data:

```bash
python scripts/build_synthetic_changepoint_dataset.py \
  --output-dir data_synthetic_changepoint_andi \
  --length 100 \
  --seed 42
```

Generate Protocol B data:

```bash
python scripts/build_synthetic_with_without_changepoint_dx.py \
  --output-dir data_synthetic_with_without_changepoint_dx \
  --length 100 \
  --seed 42
```

Generate the `L=200` Protocol B extension:

```bash
python scripts/build_synthetic_with_without_changepoint_dx.py \
  --output-dir data_synthetic_with_without_changepoint_dx \
  --length 200 \
  --seed 42
```

Run the reduced ConvTransformer-v2 multiseed protocol:

```bash
python scripts/run_multiseed_v2_L100.py \
  --data-dir data_synthetic_with_without_changepoint_dx \
  --results-dir results/multiseed_v2_L100_reduced10k_convtransformer_only \
  --architectures convtransformer \
  --seeds 1,2,3,42,123 \
  --epochs 20 \
  --batch-size 1024 \
  --max-train-samples 10000 \
  --subset-seed 2026
```

Inspect the PELT baseline options:

```bash
python scripts/run_pelt_baseline_L100.py --help
```

Reproduce the final PELT configuration:

```bash
python scripts/run_pelt_baseline_L100.py \
  --data-dir data_synthetic_with_without_changepoint_dx \
  --output-dir results/pelt_baseline_L100_full_penalty8 \
  --feature energy_abs \
  --window 5 \
  --min-size 10 \
  --jump 5 \
  --penalties 1,3,5,8,10,15,20,40,80 \
  --missing-localization-penalty 100
```

Inspect exploratory ConvTransformer-v3 commands:

```bash
python scripts/run_convtransformer_v3a_L100.py --help
python scripts/run_convtransformer_v3b_L100.py --help
```

For the `L=200` workflow, run notebook `12_synthetic_dataset_binary_changepoint_L200.ipynb` first to generate the data, then notebook `13_convtransformer_L200_detection_localization.ipynb` to train and evaluate the saved ConvTransformer configuration. The final saved notebook run corresponds to `TFM_RUN_MODE=GLOBAL`.

## Methodological limits

- The dataset changes model identity and anomalous exponent together, so these effects are not causally isolated.
- The main `L=100` architecture comparison is based on one execution per model.
- The ConvTransformer-v2 multiseed study uses a reduced training subset and cannot be treated as the same protocol as the main execution.
- The PELT baseline is a basic classical reference, not an exhaustive comparison against specialized changepoint or anomalous-diffusion methods.
- The `L=100` and `L=200` ConvTransformer runs differ in more than trajectory length.
- Per-trajectory standardization uses the full offline sequence.
- v3a and v3b are exploratory validation-stage variants without final test metrics.
- Results are synthetic and do not establish performance on experimental trajectories.

## Author

Soulaiman Chair
