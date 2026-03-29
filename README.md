# SPY Stock Price Prediction with LSTM

**Time-Series Forecasting using Deep Learning on QuantConnect Research Environment**

A Jupyter notebook project that builds and evaluates an **LSTM neural network** to predict the next-day closing price of the **SPY ETF** (S&P 500 tracker). Developed as part of coursework in **Big Data and Data Mining**.

---

## 🎯 Project Overview

This project demonstrates the application of **Recurrent Neural Networks (LSTM)** for financial time-series forecasting. Using historical daily closing prices of SPY, the model learns temporal patterns to predict future prices while handling market volatility and non-stationary data.

The notebook was built and tested in the **QuantConnect Research** environment (QuantBook), which provides clean access to high-quality financial data.

### Key Results (from the cleaned implementation)
- **Lookback window**: 90 days
- **Training period**: 2015–2021
- **Test period**: 2022–June 2025 (out-of-sample)
- **Model**: Stacked LSTM with Dropout regularization
- **Evaluation Metric**: Test RMSE ≈ **$12 per day** (as achieved in earlier experiments)

---

## 📊 Features & Methodology

- Data retrieval using **QuantConnect QuantBook** (`qb.History`)
- Proper train/test split by date (avoiding look-ahead bias)
- Min-Max scaling fitted only on training data
- Sequence creation with sliding window (90-day lookback)
- LSTM architecture:
  - 3 stacked LSTM layers (50 units each)
  - Dropout (0.2) for regularization
  - Dense output layer for next-day price prediction
- Early stopping to prevent overfitting
- Inverse scaling and visualization of actual vs. predicted prices

---

## 🛠️ Tech Stack

- **Environment**: QuantConnect Research (Jupyter Notebook)
- **Data**: QuantBook + pandas
- **Modeling**: Keras / TensorFlow (LSTM layers)
- **Preprocessing**: scikit-learn `MinMaxScaler`, NumPy
- **Visualization**: Matplotlib
- **Other**: EarlyStopping callback

---

## 📁 Notebook Structure

The provided `research.ipynb` contains:

1. **Basic QuantConnect Demo** – Bollinger Bands indicator on SPY
2. **Initial LSTM Implementation** – Early version with some data handling
3. **Improved LSTM Version** (Recommended):
   - Cleaner data pipeline
   - Fixed train/test split by date
   - 90-day sequence length
   - Validation split + early stopping
   - Proper RMSE calculation on 2022–2025 test set
   - Clear actual vs. predicted plot

---

## 🚀 How to Run

### QuantConnect Cloud (Easiest)
1. Upload the notebook to a new **Research** project on [QuantConnect.com](https://www.quantconnect.com)
