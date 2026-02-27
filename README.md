# Stock Forecast Multi-Model Pipeline

This project is a stock price forecasting application utilizing 5 different time-series models for three selected Indonesian stock tickers.

## Supported Stock Tickers
- **BUMI** (Bumi Resources Tbk)
- **ELSA** (Elnusa Tbk)
- **DEWA** (Darma Henwa Tbk)

## Supported Models
The application implements and compares the following 5 forecasting models:
1. **Prophet**: Additive regression model handling trend, seasonality, and holidays.
2. **Hybrid Prophet + XGBoost**: Ensemble approach using Prophet for base forecast and XGBoost for residual correction.
3. **N-HiTS (Neural Hierarchical Interpolation for Time Series)**: Deep learning model with multi-rate signal sampling.
4. **NeuralProphet**: Neural network-based implementation of Prophet.
5. **N-BEATS (Neural Basis Expansion Analysis for Time Series)**: Deep learning model using residual stacking for univariate series.

## Project Structure
- `app.py`: Main Streamlit web application.
- `artifacts/`: Contains data loading, feature engineering, and model prediction logic.
- `artifacts/notebooks/`: Jupyter notebooks used for training the models offline.
- `models/`: Directory containing the exported, trained model artifacts (`.joblib`, `.darts`).
- `scripts/`: Utility scripts for validation, testing, and re-running model training.

## Usage
To run the Streamlit application:
```bash
pip install -r requirements.txt
streamlit run app.py
```
