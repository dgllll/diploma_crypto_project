# Diploma Crypto Trading System

A research project for BTC price prediction and backtesting trading strategies on historical market data.

## What This Project Does

- loads Bitcoin OHLCV data from Yahoo Finance, Binance, or Coinbase;
- preprocesses the data and builds technical features;
- trains an ensemble of models for price prediction;
- evaluates prediction quality;
- runs backtests for trading strategies, including more aggressive variants.

By default, the configuration uses `BTC-USD` with a `1h` timeframe.

## Structure

- `main.py` - the basic end-to-end pipeline: data loading, feature engineering, training, evaluation, and simple backtesting.
- `main_aggresive.py` - the aggressive trading version.
- `main_aggresive_SL.py` - the aggressive version with stop-loss logic.
- `main_stoploss.py` - the most advanced scenario: short positions, risk management, and multi-timeframe analysis.
- `data/` - data loading, preprocessing, and saved CSV snapshots.
- `features/` - technical indicators, lags, and feature engineering.
- `models/` - prediction pipelines, ensembles, optimization, and saved models.
- `trading/` - trading strategies and basic backtesting.
- `evaluation/` - metrics and visualizations.
- `results/` - generated reports and run outputs.
- `another_versions/` - archived and experimental script variants.

## Models and Logic

The project uses gradient boosting and ensemble models, including `XGBoost`, `LightGBM`, and `HistGradientBoostingRegressor`. Features are built from OHLCV data, technical indicators, and lagged values.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

To run other scenarios:

```bash
python main_aggresive.py
python main_aggresive_SL.py
python main_stoploss.py
```

## Output Files

- raw data is saved in `data/`;
- trained models are saved in `models/saved/`;
- charts and reports are saved in `results/`.

## Note

The repository already contains many generated artifacts such as `data/*.csv` and `models/saved/*.pkl`, so the project is fairly large and the first `clone` or `push` may take a while.
