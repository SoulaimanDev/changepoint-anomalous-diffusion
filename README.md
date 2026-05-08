# Changepoint Detection in Anomalous Diffusion Trajectories

This repository presents a deep learning study for detecting changepoints in synthetic anomalous diffusion trajectories. The project focuses on one-dimensional trajectories composed of two consecutive segments generated from different diffusion models. The main objective is to estimate the temporal position where the underlying diffusion dynamics change.

The work is developed as an academic machine learning project at Master level. It combines synthetic data generation, model training, transition-wise evaluation and final comparison between several neural architectures.

## Project Overview

Anomalous diffusion appears in many physical, biological and complex systems where the motion of a particle does not follow classical Brownian behavior. In this project, each trajectory has a fixed length of `L = 100` and contains one changepoint separating two different diffusion regimes.

The considered diffusion models are:

- `ATTM` — Annealed Transient Time Motion
- `CTRW` — Continuous-Time Random Walk
- `FBM` — Fractional Brownian Motion
- `LW` — Lévy Walk
- `SBM` — Scaled Brownian Motion

Since there are five models and the same model is not repeated before and after the changepoint, the dataset contains:

```text
5 × 4 = 20 ordered transitions
```

Examples:

```text
ATTM → CTRW
CTRW → LW
FBM → SBM
SBM → ATTM
```

The task is formulated as a regression problem: given a trajectory, the model predicts the normalized changepoint position. The prediction is then converted back to temporal points in order to compute MAE and RMSE.

## Dataset Design

The dataset was generated in a balanced way across all ordered transitions.

| Split | Trajectories per transition | Number of transitions | Total trajectories |
|---|---:|---:|---:|
| Training | 10,000 | 20 | 200,000 |
| Validation | 1,000 | 20 | 20,000 |
| Test | 10,000 | 20 | 200,000 |

Main configuration:

```text
Trajectory length: L = 100
Minimum segment length: 20
Number of diffusion models: 5
Number of ordered transitions: 20
Training set: 200,000 trajectories
Validation set: 20,000 trajectories
Test set: 200,000 trajectories
```

## Models Studied

Four neural architectures were trained and evaluated under the same dataset protocol.

### LSTM

The LSTM model treats each trajectory as a temporal sequence. It uses recurrent layers to capture dependencies along time and to summarize the trajectory before predicting the changepoint position.

### ConvLSTM

The ConvLSTM model combines local pattern extraction with recurrent memory. The trajectory is represented using local windows, allowing the model to analyze short-range variations while preserving temporal information between consecutive regions.

### Transformer

The Transformer model uses a learnable positional encoding and multi-head self-attention. This allows the model to compare different temporal positions within the same trajectory and identify changes between the first and second segment.

### ConvTransformer

The ConvTransformer combines local feature extraction and attention-based sequence modeling. The input includes several channels derived from the trajectory, such as the position signal and local increment-based features. This helps the model capture both local variations and longer temporal relations.

## Repository Structure

```text
changepoint-anomalous-diffusion/
│
├── data_synthetic_changepoint_andi/
│   ├── dataset_summary.json
│   ├── train_pair_counts.csv
│   ├── val_pair_counts.csv
│   └── test_pair_counts.csv
│
├── notebooks/
│   ├── 01_synthetic_dataset_changepoint_anomalous_diffusion.ipynb
│   ├── 02_lstm_changepoint_detection.ipynb
│   ├── 03_convlstm_changepoint_detection.ipynb
│   ├── 04_transformer_changepoint_detection.ipynb
│   ├── 05_convtransformer_changepoint_detection.ipynb
│   └── 06_model_validation_changepoint_comparison.ipynb
│
├── scripts/
│   └── build_synthetic_changepoint_dataset.py
│
├── .gitignore
└── README.md
```

## Notebook Description

| Notebook | Purpose |
|---|---|
| `01_synthetic_dataset_changepoint_anomalous_diffusion.ipynb` | Generates and verifies the synthetic changepoint dataset. |
| `02_lstm_changepoint_detection.ipynb` | Trains and evaluates the LSTM model. |
| `03_convlstm_changepoint_detection.ipynb` | Trains and evaluates the ConvLSTM model. |
| `04_transformer_changepoint_detection.ipynb` | Trains and evaluates the Transformer model. |
| `05_convtransformer_changepoint_detection.ipynb` | Trains and evaluates the ConvTransformer model. |
| `06_model_validation_changepoint_comparison.ipynb` | Compares the four trained architectures using global and transition-wise metrics. |

## Evaluation Metrics

The models are evaluated using two regression metrics:

```text
MAE = mean(|predicted_cp - true_cp|)
RMSE = sqrt(mean((predicted_cp - true_cp)^2))
```

Both metrics are reported in temporal points of the trajectory.

## Final Results

| Model | MAE | RMSE |
|---|---:|---:|
| ConvTransformer | 7.20 | 11.63 |
| LSTM | 7.72 | 11.71 |
| ConvLSTM | 8.71 | 12.58 |
| Transformer | 11.41 | 15.19 |

The ConvTransformer obtained the lowest MAE, followed closely by the LSTM. The Transformer showed the highest global error in this experiment.

## Transition-wise Analysis

The results show that changepoint localization difficulty depends strongly on the ordered pair of diffusion models.

Easier transitions:

```text
CTRW → LW
LW → CTRW
ATTM → LW
LW → ATTM
```

More difficult transitions:

```text
SBM → FBM
FBM → SBM
SBM → ATTM
ATTM → SBM
```

This suggests that some diffusion regimes produce clearer temporal changes, while others generate trajectories where the boundary between the two segments is less easy to identify.

## Main Findings

- A balanced transition-wise dataset is important to evaluate changepoint detection fairly.
- LSTM-based models provide strong performance for short anomalous diffusion trajectories.
- ConvTransformer achieved the best global MAE in this study.
- Transformer attention alone was less effective than the hybrid or recurrent approaches in this configuration.
- Errors vary significantly depending on the ordered transition between diffusion models.
- The most difficult cases are associated with transitions where the two regimes have similar or less clearly separable temporal behavior.

## How to Use

Clone the repository:

```bash
git clone https://github.com/SoulaimanDev/changepoint-anomalous-diffusion.git
cd changepoint-anomalous-diffusion
```

Install the main Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow tqdm
```

Recommended notebook order:

```text
notebooks/01_synthetic_dataset_changepoint_anomalous_diffusion.ipynb
notebooks/02_lstm_changepoint_detection.ipynb
notebooks/03_convlstm_changepoint_detection.ipynb
notebooks/04_transformer_changepoint_detection.ipynb
notebooks/05_convtransformer_changepoint_detection.ipynb
notebooks/06_model_validation_changepoint_comparison.ipynb
```

## Reproducibility Notes

The final experiments were run using the full dataset configuration:

```text
FAST_RUN = False
Training trajectories = 200,000
Validation trajectories = 20,000
Test trajectories = 200,000
```

For reproducibility, dataset paths should be adapted to the local environment. The recommended structure is to keep the folder `data_synthetic_changepoint_andi/` at the root of the project.



