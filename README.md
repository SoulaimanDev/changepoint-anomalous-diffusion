\# Synthetic Changepoint Dataset for Anomalous Diffusion



This repository documents the construction and validation of a synthetic dataset for changepoint localization in anomalous diffusion trajectories.



Each trajectory has fixed length `L = 100` and contains a single changepoint separating two segments generated from different diffusion models. The considered models are ATTM, CTRW, FBM, LW, and SBM.



\## Dataset design



\- Trajectory length: 100

\- Minimum segment length: 20

\- Ordered transitions: 5 × 4 = 20

\- Training set: 20 × 10,000 = 200,000 trajectories

\- Validation set: 20 × 1,000 = 20,000 trajectories

\- Test set: 20 × 10,000 = 200,000 trajectories



\## Repository structure



```text

notebooks/

└── 01\_synthetic\_dataset\_changepoint\_anomalous\_diffusion.ipynb



scripts/

└── build\_synthetic\_changepoint\_dataset.py



data\_synthetic\_changepoint\_andi/

├── dataset\_summary.json

├── train\_pair\_counts.csv

├── val\_pair\_counts.csv

└── test\_pair\_counts.csv

