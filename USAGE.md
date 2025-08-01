# Forex Trading AI Agent - Usage Guide

## Quick Start

### 1. Installation

First, install the required dependencies:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install TA-Lib (required for technical indicators)
# On Ubuntu/Debian:
sudo apt-get install libta-lib-dev

# On macOS:
brew install ta-lib

# On Windows:
# Download and install from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
```

### 2. Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
# Edit .env with your API keys and preferences
```

### 3. Test Installation

Run the installation test to verify everything is working:

```bash
python3 test_installation.py
```

### 4. Demo Mode

Run the system in demo mode to see it in action with sample data:

```bash
python3 main.py --mode demo
```

## Detailed Usage

### Training Models

Train ML models on historical forex data:

```bash
# Train Random Forest model on EURUSD 1H data
python3 main.py --mode train --symbol EURUSD --timeframe 1H --models rf

# Train both LSTM and Random Forest models
python3 main.py --mode train --symbol EURUSD --timeframe 1H --models lstm rf

# Train with hyperparameter tuning
python3 main.py --mode train --symbol EURUSD --timeframe 1H --models rf --hyperparameter-tuning

# Custom date range and parameters
python3 main.py --mode train \
    --symbol GBPUSD \
    --timeframe 4H \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --models rf lstm \
    --prediction-horizon 12 \
    --epochs 100 \
    --plot-importance
```

### Command Line Options

#### Mode Selection
- `--mode`: Operation mode (`train`, `backtest`, `live`, `demo`)

#### Data Parameters
- `--symbol`: Currency pair symbol (default: EURUSD)
- `--timeframe`: Timeframe (1H, 4H, 1D, etc.)
- `--start-date`: Start date (YYYY-MM-DD format)
- `--end-date`: End date (YYYY-MM-DD format)

#### Model Parameters
- `--models`: Models to train (lstm, rf, xgb, lgb)
- `--prediction-horizon`: Prediction horizon in time periods
- `--sequence-length`: Sequence length for LSTM models
- `--feature-selection-k`: Number of features to select

#### Training Parameters
- `--epochs`: Number of training epochs for neural networks
- `--batch-size`: Batch size for training
- `--hyperparameter-tuning`: Enable hyperparameter optimization

#### Output Parameters
- `--plot-importance`: Plot feature importance
- `--save-plots`: Save plots to files

## Examples

### Example 1: Basic Training

Train a Random Forest model on EURUSD hourly data:

```bash
python3 main.py --mode train \
    --symbol EURUSD \
    --timeframe 1H \
    --models rf \
    --start-date 2022-01-01 \
    --end-date 2023-12-31
```

### Example 2: Advanced Training

Train multiple models with custom parameters:

```bash
python3 main.py --mode train \
    --symbol GBPUSD \
    --timeframe 4H \
    --models rf lstm \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --prediction-horizon 24 \
    --sequence-length 120 \
    --feature-selection-k 30 \
    --epochs 50 \
    --hyperparameter-tuning \
    --plot-importance
```

### Example 3: Demo Mode

Run the system in demo mode to see it working with sample data:

```bash
python3 main.py --mode demo
```

## Configuration

The system uses environment variables for configuration. Key settings include:

### Data Providers
```env
ALPHA_VANTAGE_API_KEY=your_api_key
OANDA_API_KEY=your_api_key
OANDA_ACCOUNT_ID=your_account_id
```

### Trading Configuration
```env
INITIAL_BALANCE=10000.0
MAX_RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.05
CURRENCY_PAIRS=EURUSD,GBPUSD,USDJPY,AUDUSD
```

### Machine Learning
```env
MODEL_RETRAIN_INTERVAL=24
LOOKBACK_PERIOD=100
PREDICTION_HORIZON=24
FEATURE_SELECTION=true
ENSEMBLE_MODELS=lstm,rf,xgb,lgb
```

## Supported Currency Pairs

The system supports major forex pairs:
- EURUSD - Euro/US Dollar
- GBPUSD - British Pound/US Dollar
- USDJPY - US Dollar/Japanese Yen
- AUDUSD - Australian Dollar/US Dollar
- USDCAD - US Dollar/Canadian Dollar
- USDCHF - US Dollar/Swiss Franc
- NZDUSD - New Zealand Dollar/US Dollar

## Supported Timeframes

- 1H - 1 Hour
- 4H - 4 Hours
- 1D - 1 Day
- 1W - 1 Week
- 1M - 1 Month

## Machine Learning Models

### Random Forest (rf)
- Fast training and prediction
- Good interpretability with feature importance
- Robust to overfitting
- No need for feature scaling

### LSTM Neural Network (lstm)
- Captures temporal patterns in price data
- Suitable for sequential data
- Requires more computational resources
- Needs TensorFlow installation

### XGBoost (xgb) - Coming Soon
- Gradient boosting algorithm
- High performance on structured data
- Built-in feature importance

### LightGBM (lgb) - Coming Soon
- Fast gradient boosting
- Memory efficient
- Good for large datasets

## Output Files

The system creates several output files:

### Models
- `models/saved/rf_SYMBOL_TIMEFRAME.pkl` - Trained Random Forest models
- `models/saved/lstm_SYMBOL_TIMEFRAME.pkl` - Trained LSTM models
- `models/saved/lstm_SYMBOL_TIMEFRAME.h5` - LSTM model weights

### Plots
- `plots/rf_importance_SYMBOL_TIMEFRAME.png` - Feature importance plots
- `plots/training_history_SYMBOL_TIMEFRAME.png` - Training history plots

### Logs
- `logs/trading.log` - Application logs

### Data
- `data/sample/` - Sample data files

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'talib'**
   - Install TA-Lib system library first, then pip install ta-lib
   - See installation instructions in README.md

2. **No data available for symbol**
   - Check internet connection
   - Verify symbol name is correct
   - Try different date range

3. **TensorFlow errors**
   - Install TensorFlow: `pip install tensorflow`
   - For GPU support: `pip install tensorflow-gpu`

4. **Memory errors with LSTM**
   - Reduce sequence length: `--sequence-length 30`
   - Reduce batch size: `--batch-size 16`
   - Use smaller dataset date range

### Performance Tips

1. **For faster training:**
   - Use Random Forest instead of LSTM
   - Reduce feature selection: `--feature-selection-k 20`
   - Use smaller date ranges

2. **For better accuracy:**
   - Use longer training periods
   - Enable hyperparameter tuning
   - Use ensemble of multiple models
   - Increase sequence length for LSTM

3. **For production use:**
   - Set up proper data providers (OANDA, Alpha Vantage)
   - Configure risk management parameters
   - Enable monitoring and alerts
   - Use database for data storage

## Next Steps

This is a foundational implementation. To extend the system:

1. **Add more data providers** (OANDA, MetaTrader 5, Alpha Vantage)
2. **Implement backtesting framework**
3. **Add risk management system**
4. **Create trading execution logic**
5. **Build web dashboard**
6. **Add more ML models** (XGBoost, LightGBM)
7. **Implement ensemble methods**
8. **Add real-time monitoring**

## Support

For issues and questions:
1. Check the logs in `logs/trading.log`
2. Run `python3 test_installation.py` to verify setup
3. Review the configuration in `.env`
4. Check the documentation in `README.md`