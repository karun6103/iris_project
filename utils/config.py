"""
Configuration management for the Forex Trading AI Agent.
Handles loading and validation of environment variables.
"""

import os
from typing import List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DataProviderConfig:
    """Configuration for data providers."""
    alpha_vantage_api_key: Optional[str] = None
    oanda_api_key: Optional[str] = None
    oanda_account_id: Optional[str] = None
    mt5_login: Optional[str] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None

@dataclass
class TradingConfig:
    """Configuration for trading parameters."""
    initial_balance: float = 10000.0
    max_risk_per_trade: float = 0.02
    max_daily_loss: float = 0.05
    max_positions: int = 5
    currency_pairs: List[str] = None
    timeframes: List[str] = None

@dataclass
class RiskConfig:
    """Configuration for risk management."""
    stop_loss_percentage: float = 0.02
    take_profit_ratio: float = 2.0
    max_drawdown: float = 0.15
    position_size_method: str = "kelly"

@dataclass
class MLConfig:
    """Configuration for machine learning models."""
    model_retrain_interval: int = 24
    lookback_period: int = 100
    prediction_horizon: int = 24
    feature_selection: bool = True
    ensemble_models: List[str] = None

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    commission: float = 0.0001
    spread: float = 0.0002

@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    enable_notifications: bool = True

@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    database_url: Optional[str] = None
    redis_url: str = "redis://localhost:6379/0"

@dataclass
class APIConfig:
    """Configuration for API server."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    file: str = "logs/trading.log"
    max_size: str = "10MB"
    backup_count: int = 5

@dataclass
class Config:
    """Main configuration class that combines all configuration sections."""
    data_providers: DataProviderConfig
    trading: TradingConfig
    risk: RiskConfig
    ml: MLConfig
    backtest: BacktestConfig
    notifications: NotificationConfig
    database: DatabaseConfig
    api: APIConfig
    logging: LoggingConfig
    debug: bool = False
    testing: bool = False

def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def get_env_list(key: str, default: List[str] = None, separator: str = ',') -> List[str]:
    """Get list value from environment variable."""
    if default is None:
        default = []
    value = os.getenv(key, '')
    if not value:
        return default
    return [item.strip() for item in value.split(separator) if item.strip()]

def load_config() -> Config:
    """Load configuration from environment variables."""
    
    # Data providers configuration
    data_providers = DataProviderConfig(
        alpha_vantage_api_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
        oanda_api_key=os.getenv('OANDA_API_KEY'),
        oanda_account_id=os.getenv('OANDA_ACCOUNT_ID'),
        mt5_login=os.getenv('MT5_LOGIN'),
        mt5_password=os.getenv('MT5_PASSWORD'),
        mt5_server=os.getenv('MT5_SERVER')
    )
    
    # Trading configuration
    trading = TradingConfig(
        initial_balance=float(os.getenv('INITIAL_BALANCE', '10000.0')),
        max_risk_per_trade=float(os.getenv('MAX_RISK_PER_TRADE', '0.02')),
        max_daily_loss=float(os.getenv('MAX_DAILY_LOSS', '0.05')),
        max_positions=int(os.getenv('MAX_POSITIONS', '5')),
        currency_pairs=get_env_list('CURRENCY_PAIRS', ['EURUSD', 'GBPUSD', 'USDJPY']),
        timeframes=get_env_list('TIMEFRAMES', ['1H', '4H', '1D'])
    )
    
    # Risk management configuration
    risk = RiskConfig(
        stop_loss_percentage=float(os.getenv('STOP_LOSS_PERCENTAGE', '0.02')),
        take_profit_ratio=float(os.getenv('TAKE_PROFIT_RATIO', '2.0')),
        max_drawdown=float(os.getenv('MAX_DRAWDOWN', '0.15')),
        position_size_method=os.getenv('POSITION_SIZE_METHOD', 'kelly')
    )
    
    # Machine learning configuration
    ml = MLConfig(
        model_retrain_interval=int(os.getenv('MODEL_RETRAIN_INTERVAL', '24')),
        lookback_period=int(os.getenv('LOOKBACK_PERIOD', '100')),
        prediction_horizon=int(os.getenv('PREDICTION_HORIZON', '24')),
        feature_selection=get_env_bool('FEATURE_SELECTION', True),
        ensemble_models=get_env_list('ENSEMBLE_MODELS', ['lstm', 'rf', 'xgb', 'lgb'])
    )
    
    # Backtesting configuration
    backtest = BacktestConfig(
        start_date=os.getenv('BACKTEST_START_DATE', '2020-01-01'),
        end_date=os.getenv('BACKTEST_END_DATE', '2023-12-31'),
        commission=float(os.getenv('COMMISSION', '0.0001')),
        spread=float(os.getenv('SPREAD', '0.0002'))
    )
    
    # Notifications configuration
    notifications = NotificationConfig(
        telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
        telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID'),
        enable_notifications=get_env_bool('ENABLE_NOTIFICATIONS', True)
    )
    
    # Database configuration
    database = DatabaseConfig(
        database_url=os.getenv('DATABASE_URL'),
        redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    )
    
    # API configuration
    api = APIConfig(
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', '8000')),
        workers=int(os.getenv('API_WORKERS', '4'))
    )
    
    # Logging configuration
    logging_config = LoggingConfig(
        level=os.getenv('LOG_LEVEL', 'INFO'),
        file=os.getenv('LOG_FILE', 'logs/trading.log'),
        max_size=os.getenv('LOG_MAX_SIZE', '10MB'),
        backup_count=int(os.getenv('LOG_BACKUP_COUNT', '5'))
    )
    
    return Config(
        data_providers=data_providers,
        trading=trading,
        risk=risk,
        ml=ml,
        backtest=backtest,
        notifications=notifications,
        database=database,
        api=api,
        logging=logging_config,
        debug=get_env_bool('DEBUG', False),
        testing=get_env_bool('TESTING', False)
    )

# Global configuration instance
config = load_config()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config

def reload_config() -> Config:
    """Reload configuration from environment variables."""
    global config
    config = load_config()
    return config