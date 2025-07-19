# CNN-LSTM Climate Emulator 🌎

This project presents a hybrid CNN–LSTM architecture for emulating expensive climate simulations using deep learning. Built for CSE 151B (Deep Learning) at UC San Diego.

## 🚀 Overview
Traditional climate models are computationally expensive. We built a fast, high-fidelity surrogate model that predicts climate variables—like surface temperature (tas) and precipitation (pr)—using 12-month sequences of historical forcings (e.g., CO2, CH4).

### ✅ Architecture Highlights
- Residual CNN blocks with GroupNorm & SiLU activation
- Multi-layer unidirectional LSTM with Multi-Head Attention
- Optuna for hyperparameter tuning
- Trained with PyTorch Lightning, logged with Weights & Biases

## 🧠 Key Results
- **Public Kaggle Score**: 0.6965  
- **Private Kaggle Score**: 0.7776 (Ranked 5th overall)

## 📁 Files
- `report.pdf`: Final research report
- `model/`: Model architecture and training code
- `configs/`: Hydra configs
- `train.py`, `eval.py`: Training & evaluation scripts

## 📊 Dataset
- Monthly gridded climate data from SSP126, SSP370, SSP585
- Predicts future SSP245 scenario

## 🧑‍💻 Author
Devam Derasary 
_CSE 151B, Spring 2025, UC San Diego_

## 🔗 Citation
The full final report will be uploaded to this repository soon. Once available, please cite it if referencing this work.


