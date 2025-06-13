# Bitcoin Price Prediction: XGBoost vs LSTM

This repository presents two machine learning approaches to predict Bitcoin prices using historical 30-minute interval data. The models use different learning techniques — a tree-based gradient boosting model (XGBoost) and a deep learning model (LSTM) — to estimate future Bitcoin prices based on technical indicators and time series features.

## Project Overview

- **Goal**: Compare classical and deep learning approaches for predicting short-term Bitcoin closing prices.
- **Data**: OHLCV data with 30-minute resolution.
- **Techniques Used**:
  - XGBoost Regressor with engineered features and grid search tuning.
  - LSTM Neural Network with sequence data and normalized scaling.
- **Evaluation**: R² score, MAE, RMSE, and visual inspection.

---

## Repository Contents

### 1. `Bitcoin_XGBoost_model.ipynb`
A machine learning pipeline using XGBoost:

- Feature engineering: lag features, moving averages, EMA, and volatility
- Train/test split preserving temporal order
- GridSearchCV for hyperparameter optimization
- Model performance evaluation and prediction plotting

**Best R²**: ~0.96  
**Best MAE**: ~1161.68

### 2. `Bitcoin_LSTM_model.ipynb`
A deep learning pipeline using Long Short-Term Memory (LSTM) neural networks:

- Time series normalization and windowed sequence generation
- Sequence-based input suitable for LSTM layers
- Model built with TensorFlow/Keras
- Training history and performance plots
