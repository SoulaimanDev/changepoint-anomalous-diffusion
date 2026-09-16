# Changepoint Detection in Anomalous Diffusion

Historical experiment code and verified tuning and validation-calibration exports for the September 2026 Master's thesis on changepoint detection and temporal localization in synthetic anomalous-diffusion trajectories.

The scientific objective is deliberately narrow: given a one-dimensional trajectory generated from anomalous-diffusion processes, decide whether it contains one model transition and, when a transition is present, estimate its temporal position. The repository does not claim to identify the diffusion model or infer the anomalous exponent inside each segment.

## Timeline and reproducibility scope

The reference scientific manuscript was closed on **8 September 2026**. The thesis documents the experimental development up to submission. The complete confirmatory tuning campaign reported here was completed and published afterwards, on **14 September 2026**, as a post-submission update; its 450-run results are not in that earlier PDF.

- **Historical experiments from the thesis:** retained results and their scripts/notebooks, under the individual historical protocols.
- **Completed confirmatory tuning:** 450 validation-based runs and the selected configurations, with a lightweight script to validate the published exports and reproduce their summaries.
- **Completed final training / validation-calibration:** the 16 September 2026 backup contains 25/25 runs with seeds `1, 2, 3, 42, 123`; sealed calibration metrics are now published separately.
- **Independent held-out test:** a separate subsequent phase, with no data or results included in this update.

Reproduction has three distinct scopes. The published tuning exports can be checked locally without training or datasets. Historical experiments have scripts/notebooks, but require their data and compatible dependencies. The exact confirmatory campaign code, frozen configurations and environment are retained in the separately verified scientific archive and are not fully exposed in `main`; this public repository alone does not currently reproduce all 450 training runs. See [provenance](results/tuning/PROVENANCE.md).

## Complete tuning campaign — 450 runs

This campaign addresses asymmetries in the earlier comparisons: the search spaces were designed to be more comparable under a common frozen selection protocol, with 5 architectures, 30 candidate configurations per architecture, tuning seeds `11, 29, 47`, and 450 runs in total. Selection uses validation only and the same frozen criterion order for every architecture: maximize mean F1, then minimize mean MAE_all, then minimize mean FPR among eligible configurations. The independent test is not used. Equal candidate budgets do not imply equal parameter counts, identical hyperparameters or mathematically equivalent search spaces.

The verified backup dated 14 September 2026 contains **450/450 completed tuning runs, 0 failed**: LSTM, xLSTM, CNN-LSTM, Transformer and ConvTransformer; 30 configurations per architecture and three tuning seeds `11, 29, 47`. See [complete tuning results and methodology](results/tuning/README.md) and [provenance](results/tuning/PROVENANCE.md).

| Architecture | Selected config | Mean validation F1 | Mean MAE_all | Mean FPR | Valid seeds |
|---|---|---:|---:|---:|---:|
| LSTM | cfg_02 | 0.8006514266130286 | 9.60602887471517 | 0.19066666666666668 | 3 |
| xLSTM | cfg_09 | 0.7940252554167891 | 9.782986640930176 | 0.19166666666666665 | 3 |
| CNN-LSTM | cfg_06 | 0.800410476840066 | 7.6136603355407715 | 0.19226666666666667 | 3 |
| Transformer | cfg_06 | 0.8022087901770522 | 7.887541135152181 | 0.18613333333333335 | 3 |
| ConvTransformer | cfg_12 | 0.8097509375627459 | 7.431063175201416 | 0.19766666666666666 | 3 |

ConvTransformer achieved the highest mean validation F1 among the evaluated configurations under the frozen tuning protocol. This does not prove general superiority, constitute an independent final performance estimate, or by itself constitute a statistical significance test.

These are **validation/tuning results**, used to select one configuration per architecture. They are **not final test results**. The subsequent [25-run training/calibration results](results/final_validation_2026/README.md) are published separately; independent test evaluation is not included. No final test data were used during tuning.

The earlier [campaign record](docs/confirmatory_campaign_2026.md), [infrastructure documentation](docs/runpod_reproducibility.md) and [interim results directory](results/confirmatory_2026/README.md) preserve historical context; earlier progress snapshots are superseded by the completed tuning state above. Historical experimental results below remain unchanged.

### Five confirmatory architectures

| Architecture | Campaign model family |
|---|---|
| LSTM | Unidirectional recurrent sequence model with temporal LSTM layers. |
| xLSTM | Local mLSTM-only adaptation originating from Beck et al. (2024), using matrix memory; not an exact reproduction of the official implementation or an official optimized kernel. |
| CNN-LSTM | Conv1D feature extraction followed by LSTM layers, not a ConvLSTM recurrent cell. |
| Transformer | Learned positional encoding and Transformer encoder blocks. |
| ConvTransformer | Convolutional feature extraction followed by Transformer encoder blocks in the archived confirmatory implementation. |

These five campaign architectures are distinct from the four historical implementations below. In particular, historical ConvTransformer-v2, exploratory v3a/v3b and the confirmatory ConvTransformer are not interchangeable implementations. The archive defines the exact campaign configurations. xLSTM extends the comparison beyond the historical LSTM and attention-based baselines by including a matrix-memory recurrent alternative motivated by Beck et al. (2024); its inclusion is not a claim of improved performance.

## Completed final training / validation-calibration — 25 runs

The verified backup dated **16 September 2026** contains **25/25 completed runs, 0 failed**, using the five configurations selected above and seeds `1, 2, 3, 42, 123` for each. Thresholds were calibrated on validation_calibration and frozen. **These are NOT independent test results.** The independent held-out test is a separate subsequent phase and is excluded.

| Architecture | Configuration | Seeds | F1 mean ± std | MAE_all mean ± std | FPR mean ± std |
|---|---|---:|---:|---:|---:|
| LSTM | cfg_02 | 5 | 0.791694 ± 0.002988 | 9.215873 ± 0.079549 | 0.194040 ± 0.003057 |
| xLSTM | cfg_09 | 5 | 0.788195 ± 0.005167 | 9.690845 ± 0.184891 | 0.188920 ± 0.006128 |
| CNN-LSTM | cfg_06 | 5 | 0.786514 ± 0.003121 | 7.599934 ± 0.055970 | 0.186640 ± 0.003812 |
| Transformer | cfg_06 | 5 | 0.792640 ± 0.005474 | 7.640302 ± 0.166553 | 0.191920 ± 0.004621 |
| ConvTransformer | cfg_12 | 5 | 0.799563 ± 0.005949 | 7.294523 ± 0.238058 | 0.195240 ± 0.003971 |

Mean ± sample standard deviation (`ddof=1`, five seeds), computed from the sealed validation-calibration metrics. ConvTransformer obtained the highest observed mean validation-calibration F1 under this protocol; this does not establish general superiority, statistical significance or independent-test generalization. See [per-run metrics, provenance and validation](results/final_validation_2026/README.md). The preceding tuning table is unchanged and uses a different validation split and seed set.

## Essential scientific references

This project is built around the AnDi ecosystem, but it uses a thesis-specific supervised protocol rather than the original challenge tasks.

- Muñoz-Gil et al. (2021), DOI `10.1038/s41467-021-26320-w`, provides the AnDi context and the distinction between T1 anomalous-exponent inference, T2 diffusion-model classification, and T3 trajectory segmentation.
- Bo et al. (2019), DOI `10.1103/PhysRevE.100.010102`, and Argun et al. (2021), DOI `10.1088/1751-8121/ac070a`, provide recurrent-neural-network precedents for anomalous-diffusion characterization, including the use of recurrent models on short single trajectories.
- Garibo-i-Orts et al. (2021), DOI `10.1088/1751-8121/ac3707`, is a domain-specific precedent for convolution + BiLSTM sequence modelling.
- Vaswani et al. (2017), "Attention Is All You Need", is the general Transformer reference.
- Firbas et al. (2023), DOI `10.1088/1751-8121/acafb3`, is a domain-specific precedent for convolutional Transformer models in anomalous-diffusion characterization.

- Hochreiter and Schmidhuber (1997), "Long Short-Term Memory", DOI `10.1162/neco.1997.9.8.1735`, is the foundational LSTM reference.
- Beck et al. (2024), "xLSTM: Extended Long Short-Term Memory", NeurIPS 37, [official proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html), provides the scientific origin of xLSTM.
- Killick, Fearnhead and Eckley (2012), "Optimal Detection of Changepoints With a Linear Computational Cost", DOI `10.1080/01621459.2012.737745`, is the PELT reference.

- Muñoz-Gil et al. (2025), "Quantitative evaluation of methods to analyze motion changes in single-particle experiments", DOI [10.1038/s41467-025-61949-x](https://doi.org/10.1038/s41467-025-61949-x), provides the experimental SPT and motion-change analysis context.
- Arévalo et al. (2024), "Stock volatility as an anomalous diffusion process", DOI [10.3934/math.20241663](https://doi.org/10.3934/math.20241663), provides interdisciplinary financial context only.

These are essential references for reading this repository. The complete bibliography is in the reference thesis closed on 8 September 2026; the manuscript is maintained separately from this repository.

The thesis CNN-LSTM and ConvTransformer models are adaptations for binary changepoint detection and localization. They are not exact reproductions of the Garibo-i-Orts or Firbas architectures.

## Historical experiments from the thesis

The data descriptions, architecture details, result tables and exploratory studies below describe the historical experiments. Their numbers and experimental settings are preserved; they are not results or launch instructions for the new 450-run confirmatory campaign.

### Scope: 1D trajectories and trajectory length

The synthetic 1D setting provides a controlled framework with known generating mechanisms and changepoint locations. `L=100` is the main thesis regime; `L=200` is a historical extension. These lengths support a controlled methodological study of detection and localization with observations on both sides of the change, rather than establishing a minimum length or guaranteed statistical stability.

In biological single-particle tracking (SPT), heterogeneous motion and anomalous diffusion motivate segmentation and changepoint detection, while experimental noise complicates interpretation (Muñoz-Gil et al., 2025; see the essential references). Experimental trajectories can be much shorter and noisier; sequences of approximately 10–50 steps are a harder target for future evaluation, not a regime evaluated here or a universal description of SPT data (see also Garibo-i-Orts et al., 2021, on short and noisy trajectories). The 1D, `L=100–200` regime should therefore be interpreted as a controlled intermediate setting toward shorter and noisier experimental trajectories, not a complete representation of real biological data. Here `L` counts sampled positions, so there are `L-1` increments.

Regime changes and non-stationary time series are also relevant beyond biology, including the financial anomalous-diffusion context discussed in the thesis (Arévalo et al., 2024). This is interdisciplinary motivation only; this repository reports no economic experiment and applies neither TimeGAN nor a financial model.

### Data and protocols

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

For a trajectory of length `L`, the increment sequence has length `m = L - 1`. The localization output has `m` components, one per increment index `j = 0, ..., m-1`; the admissibility mask restricts changepoints to the central positions specified above. Localization index `j` corresponds to trajectory position `τ = j + 1`. Thus `L=100` gives `m=99` output components, with admissible indices `19..79`; the remaining components are masked, not additional admissible changepoints.

The binary neural models use per-trajectory standardized increments `dx = diff(X)` as input. This is an offline normalization: the mean and standard deviation are computed from the full observed sequence.

### Architectures

- `LSTM`: recurrent baseline with two temporal layers.
- `CNN-LSTM`: one-dimensional convolutional feature extraction followed by LSTM layers.
- `Transformer`: learned positional encoding and Transformer encoder blocks.
- `ConvTransformer-v2`: local convolutional stem and residual convolutional blocks followed by Transformer encoder blocks.

The historical notebook filenames `03_convlstm_...` and `09_convlstm_...` are kept for reproducibility. Those files implement a Conv1D -> LSTM pipeline, not a ConvLSTM recurrent cell.

All Protocol B neural networks have one binary detection output and one localization output over valid changepoint positions. Localization loss is masked for negative samples. Reported point estimates use the expectation of the predicted localization distribution.

### Main L=100 results

The following values come from the frozen single main execution stored in `results/frozen_v2/`. Localization errors labelled `all` include every truly positive trajectory, including false negatives.

| Model | Threshold | Accuracy | Precision | Recall | F1 | FPR | FNR | MAE all | RMSE all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.35 | 0.7328 | 0.7341 | 0.7300 | 0.7320 | 0.2645 | 0.2700 | 12.7331 | 16.3504 |
| CNN-LSTM | 0.25 | 0.7891 | 0.7980 | 0.7743 | 0.7860 | 0.1960 | 0.2257 | 7.6645 | 11.9183 |
| Transformer | 0.30 | 0.7712 | 0.8093 | 0.7095 | 0.7561 | 0.1672 | 0.2905 | 8.7414 | 12.5307 |
| ConvTransformer-v2 | 0.30 | 0.8037 | 0.8184 | 0.7806 | 0.7990 | 0.1732 | 0.2194 | 7.5370 | 11.5706 |

Transformer gives the lowest false-positive rate in this run, while ConvTransformer-v2 has the highest F1 and lowest MAE all among these four executions. These are single-run results, not multi-seed estimates.

### ConvTransformer-v2 multiseed study

The earlier reduced-training multiseed experiment covers ConvTransformer-v2 only; it is separate from the completed five-architecture confirmatory tuning campaign:

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

### PELT baseline

`scripts/run_pelt_baseline_L100.py` implements an elementary historical classical baseline with `ruptures`, using PELT (Killick et al., 2012).

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

### ConvTransformer-v2 at L=200

Notebook `13_convtransformer_L200_detection_localization.ipynb` and `results/L200/` are preserved from `main` as the retained historical `L=200` execution.

| Threshold | Accuracy | Precision | Recall | F1 | FPR | FNR | MAE | RMSE | nMAE | nRMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.475 | 0.852365 | 0.894449 | 0.799020 | 0.844046 | 0.094290 | 0.200980 | 9.376101 | 18.274097 | 0.0469 | 0.0914 |

The executed notebook uses positive detection class weight `1.5`, negative detection class weight `1.0`, and multitask loss weights `[2.0, 1.0]`. These parameters are part of the saved run and should not be rewritten as `1.0/1.0`.

The `L=100` versus `L=200` comparison is descriptive: length, threshold grid, batch size, and class weighting are not held under a single controlled ablation.

### ConvTransformer-v3 historical exploratory work

ConvTransformer-v3a uses six signal-derived input channels; v3b adds Gaussian soft localization targets (`sigma=1` and `sigma=2`). These are historical exploratory variants, not the current confirmatory ConvTransformer. No test metrics are reported for them, and they are not presented as better than v2. Complete trial exports are not part of this publication. See [the historical v3 protocol](docs/experiment_protocol_v3.md) for technical details.

## Repository structure

```text
.
|-- configs/                         # historical configurations, including frozen_v2
|-- data_synthetic_changepoint_andi/  # historical dataset summaries, not HDF5 data
|-- docs/
|   |-- confirmatory_campaign_2026.md
|   |-- runpod_reproducibility.md
|   |-- experiment_protocol_v2.md
|   `-- experiment_protocol_v3.md
|-- notebooks/                       # historical experiments
|-- outputs/xlstm_diagnostic/         # earlier isolated numerical diagnosis
|-- results/
|   |-- tuning/                      # complete 450-run tuning exports
|   |-- final_validation_2026/        # 25 sealed training/calibration runs, not test
|   |-- confirmatory_2026/           # superseded interim publication record
|   |-- frozen_v2/
|   |-- final_tfm_additions/          # historical thesis additions, not final 25 runs
|   |-- L200/
|   |-- multiseed_v2_L100_reduced10k_convtransformer_only/
|   `-- pelt_baseline_L100_full_penalty8/
|-- scripts/                         # historical helpers and training scripts
|   `-- summarize_tuning_results.py  # read-only export validation
|-- tests/
`-- requirements.txt                 # historical dependency ranges, not campaign lock
```

The manuscript and its full bibliography are maintained separately. The archived campaign's `confirmatory/` and `runpod/` source trees are not present in public `main`.

Generated datasets (`*.h5`), trained models, and intermediate training outputs are intentionally excluded from Git. The tracked CSV, JSON, YAML, Markdown, and PNG artifacts are the source of truth for the results shown here.

## Reproduction commands

### Lightweight validation

Validate the published tuning exports using only the Python standard library, without datasets or training (see the [Windows line-ending note](results/tuning/README.md#byte-level-hashes-and-windows-checkouts)):

```bash
python -B scripts/summarize_tuning_results.py
python -B scripts/summarize_final_validation_results.py
```

### Historical experiment environment

The following setup uses the historical dependency ranges, not the archived confirmatory lock. On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

Run the historical helper tests (they may create temporary test files):

```bash
python -m unittest discover -s tests -v
```

Inspect historical baseline and exploratory options without starting an experiment:

```bash
python scripts/run_pelt_baseline_L100.py --help
python scripts/run_convtransformer_v3a_L100.py --help
python scripts/run_convtransformer_v3b_L100.py --help
```

### Potentially expensive historical data generation and experiments

**These commands generate large datasets or run historical training/evaluation. They are optional reproduction examples, not lightweight checks or instructions for the active confirmatory campaign.** Use separate local output directories and review disk/compute requirements before execution. No experiment is needed to inspect the published tuning results.

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
  --results-dir ../historical_reproduction/multiseed_v2_L100_reduced10k_convtransformer_only \
  --architectures convtransformer \
  --seeds 1,2,3,42,123 \
  --epochs 20 \
  --batch-size 1024 \
  --max-train-samples 10000 \
  --subset-seed 2026
```

Reproduce the historical PELT penalty search:

```bash
python scripts/run_pelt_baseline_L100.py \
  --data-dir data_synthetic_with_without_changepoint_dx \
  --output-dir ../historical_reproduction/pelt_baseline_L100_full_penalty8 \
  --feature energy_abs \
  --window 5 \
  --min-size 10 \
  --jump 5 \
  --penalties 1,3,5,8,10,15,20,40,80 \
  --missing-localization-penalty 100
```

For the `L=200` workflow, run notebook `12_synthetic_dataset_binary_changepoint_L200.ipynb` first to generate the data, then notebook `13_convtransformer_L200_detection_localization.ipynb` to train and evaluate the saved ConvTransformer configuration. The retained historical notebook run corresponds to `TFM_RUN_MODE=GLOBAL`.

## Methodological limits of the historical experiments

- The dataset changes model identity and anomalous exponent together, so these effects are not causally isolated.
- The main `L=100` architecture comparison is based on one execution per model.
- Historical scoring/stopping criteria differed across architectures, introducing an additional experimental asymmetry. This limits direct comparability without invalidating those experiments and motivates the common frozen selection and stopping rules of the confirmatory campaign.
- The ConvTransformer-v2 multiseed study uses a reduced training subset and cannot be treated as the same protocol as the main execution.
- The PELT baseline is a basic classical reference, not an exhaustive comparison against specialized changepoint or anomalous-diffusion methods.
- The `L=100` and `L=200` ConvTransformer runs differ in more than trajectory length.
- Per-trajectory standardization uses the full offline sequence.
- v3a and v3b are exploratory validation-stage variants without final test metrics.
- Results are synthetic and do not establish performance on experimental trajectories.

## Future work

Future studies should evaluate shorter trajectories (approximately 10–50 steps), more realistic noise conditions and experimental SPT data. The confirmatory protocol defined in the thesis's Appendix C provides a methodological basis: multi-seed comparisons, more comparable search spaces, validation-only selection and an independent final test after decisions are frozen. Appendix C is a protocol, not a result; the 450-run tuning and 25-run final training/calibration stages are complete and published separately; the independent held-out test evaluation remains outside this update.

The completed confirmatory study may provide the basis for a future manuscript once the independent final evaluation is complete.

## Author

Soulaiman Chair
