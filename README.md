# Water Quality Forecasting under Missing Observations via Multi-Semantic Graph and Covariate Compression Enhanced Gated Recurrent Unit with Decay

## Overview
This repository contains the implementation assets for multi-station water-quality forecasting under missing observations.  The project focuses on pH forecasting with auxiliary covariates, graph-aware spatial dependency modeling, and missingness-aware temporal learning.

## Method Summary
1. Build station-level relational structure from multiple semantics (instead of relying on only one spatial criterion).
2. Encode auxiliary variables into compact informative representations to reduce redundancy and noise.
3. Fuse target values, masks, and time-gap signals in a decay-aware recurrent backbone.
4. Run multi-horizon forecasting and compare strong baselines such as ARIMA, KNN, LSTM, GNN, GRUD, and XGBoost.

## Repository Structure
- `main.py`: single configuration run entry
- `run_with_baseline.py`: multi-configuration / baseline run entry
- `trainer.py`: script execution wrapper
- `utils.py`: model script discovery utilities
- `config.py`: YAML config loader
- `config_*.yaml`: model-specific experiment settings
- `data/preprocess.py`: preprocessing utilities
- `datasets/water_dataset.mat`: water-quality dataset
- `datasets/adjacency_matrix.npy`: adjacency matrix


## Configuration Style
Configuration files follow a structured format similar to:
- `seed`
- `data` (missing rate, horizon, subgraph/sample settings)
- `model` (`name` and `params`)
- `loss`
- `training`
- `paths`
- `device`

Example config files in this repository:
- `config_model.yaml`
- `config_GNN.yaml`
- `config_lstm.yaml`
- `config_KNN.yaml`
- `config_ARIMA.yaml`
- `config_xgboost.yaml`



## Data Notes
- `water_dataset.mat` stores target and auxiliary time-series tensors.
- `adjacency_matrix.npy` stores station relationship priors used by graph-based components.
- This repository is organized as a code-focused package; generated result artifacts are intentionally excluded.

## Related Work
This project is closely related to the following repository:  
https://github.com/Xielewei/Water-Quality

