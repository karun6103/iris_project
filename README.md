# Forex Trading AI Agent

An advanced AI-powered automated trading system for forex markets with machine learning-based signal generation, risk management, and backtesting capabilities.

## Features

- **Multi-source Data Integration**: Support for multiple forex data providers (Alpha Vantage, OANDA, MetaTrader 5, Yahoo Finance)
- **Advanced ML Models**: Multiple machine learning models including LSTM, Random Forest, XGBoost, and LightGBM
- **Technical Analysis**: 50+ technical indicators using TA-Lib
- **Risk Management**: Dynamic position sizing, stop-loss, take-profit, and drawdown protection
- **Backtesting Framework**: Comprehensive backtesting with performance metrics
- **Real-time Trading**: Live trading execution with multiple broker integrations
- **Monitoring & Alerts**: Real-time monitoring with Telegram notifications
- **Web Dashboard**: FastAPI-based web interface for monitoring and control

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd forex-trading-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install TA-Lib (required for technical indicators):
```bash
# On Ubuntu/Debian
sudo apt-get install libta-lib-dev

# On macOS
brew install ta-lib

# On Windows, download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

Create a `.env` file with the following variables:

```
# Data Providers
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
OANDA_API_KEY=your_oanda_key
OANDA_ACCOUNT_ID=your_oanda_account_id
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server

# Trading Configuration
INITIAL_BALANCE=10000
MAX_RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.05
CURRENCY_PAIRS=EURUSD,GBPUSD,USDJPY,AUDUSD

# Notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Database
DATABASE_URL=postgresql://user:password@localhost/forex_trading
```

## Quick Start

1. **Backtest a strategy**:
```bash
python main.py --mode backtest --strategy ml_ensemble --pair EURUSD --start 2023-01-01 --end 2023-12-31
```

2. **Start live trading**:
```bash
python main.py --mode live --strategy ml_ensemble
```

3. **Launch web dashboard**:
```bash
python dashboard.py
```

## Project Structure

```
forex-trading-ai/
├── agents/                 # AI trading agents
│   ├── base_agent.py      # Base agent class
│   ├── ml_agent.py        # ML-based trading agent
│   └── ensemble_agent.py  # Ensemble of multiple agents
├── data/                  # Data management
│   ├── providers/         # Data provider integrations
│   ├── preprocessor.py    # Data preprocessing
│   └── storage.py         # Data storage utilities
├── models/                # Machine learning models
│   ├── lstm_model.py      # LSTM neural network
│   ├── rf_model.py        # Random Forest
│   ├── xgb_model.py       # XGBoost
│   └── ensemble.py        # Model ensemble
├── strategies/            # Trading strategies
│   ├── base_strategy.py   # Base strategy class
│   ├── ml_strategy.py     # ML-based strategy
│   └── technical_strategy.py # Technical analysis strategy
├── risk/                  # Risk management
│   ├── position_sizing.py # Position sizing algorithms
│   ├── risk_manager.py    # Risk management system
│   └── drawdown.py        # Drawdown protection
├── execution/             # Trade execution
│   ├── brokers/           # Broker integrations
│   ├── executor.py        # Trade executor
│   └── order_manager.py   # Order management
├── backtesting/           # Backtesting framework
│   ├── engine.py          # Backtesting engine
│   ├── metrics.py         # Performance metrics
│   └── visualizer.py      # Results visualization
├── monitoring/            # Monitoring and alerts
│   ├── logger.py          # Logging system
│   ├── alerts.py          # Alert system
│   └── dashboard.py       # Web dashboard
├── utils/                 # Utility functions
│   ├── config.py          # Configuration management
│   ├── indicators.py      # Technical indicators
│   └── helpers.py         # Helper functions
├── main.py                # Main application entry point
├── dashboard.py           # Web dashboard server
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Usage Examples

### Backtesting
```python
from agents.ml_agent import MLTradingAgent
from backtesting.engine import BacktestEngine

# Create agent
agent = MLTradingAgent()

# Run backtest
engine = BacktestEngine()
results = engine.run_backtest(
    agent=agent,
    symbol='EURUSD',
    start_date='2023-01-01',
    end_date='2023-12-31'
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### Live Trading
```python
from agents.ml_agent import MLTradingAgent
from execution.executor import TradeExecutor

# Initialize agent and executor
agent = MLTradingAgent()
executor = TradeExecutor()

# Start live trading
agent.start_live_trading(executor)
```

## Risk Warning

**IMPORTANT**: This software is for educational and research purposes. Forex trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Please ensure you understand the risks involved and consider seeking advice from an independent financial advisor.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For questions and support, please open an issue on GitHub or contact the maintainers.