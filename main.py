#!/usr/bin/env python3
"""
Main entry point for the Forex Trading AI Agent.
Provides command-line interface for training, backtesting, and live trading.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/trading.log', mode='a')
    ]
)

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

# Import our modules
from utils.config import get_config
from data.preprocessor import DataPreprocessor
from data.providers.yfinance_provider import YFinanceProvider
from models.lstm_model import LSTMModel
from models.rf_model import RandomForestModel

async def train_models(args):
    """Train ML models on historical data."""
    logger.info("Starting model training...")
    
    config = get_config()
    
    # Initialize data preprocessor
    provider = YFinanceProvider()
    preprocessor = DataPreprocessor(provider)
    
    # Process data
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    processed_data = await preprocessor.process_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start_date,
        end_date=end_date,
        target_horizons=[args.prediction_horizon],
        feature_selection_k=args.feature_selection_k
    )
    
    logger.info(f"Processed data shape: {processed_data.shape}")
    
    # Prepare ML data
    target_column = f'direction_{args.prediction_horizon}'
    if target_column not in processed_data.columns:
        logger.error(f"Target column {target_column} not found in processed data")
        return
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data(
        processed_data, target_column, test_size=0.2
    )
    
    # Get feature names
    feature_names = X_train.columns.tolist()
    
    # Train models
    models = {}
    
    if 'lstm' in args.models:
        logger.info("Training LSTM model...")
        lstm_model = LSTMModel(sequence_length=args.sequence_length, model_type="classification")
        lstm_model.set_feature_names(feature_names)
        
        # Convert to numpy arrays for LSTM
        X_train_np = X_train.values
        X_test_np = X_test.values
        y_train_np = y_train.values
        y_test_np = y_test.values
        
        # Train LSTM
        history = lstm_model.train(
            X_train_np, y_train_np,
            X_test_np, y_test_np,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Evaluate LSTM
        lstm_metrics = lstm_model.evaluate(X_test_np, y_test_np)
        logger.info(f"LSTM Test Accuracy: {lstm_metrics['accuracy']:.4f}")
        
        # Save LSTM model
        lstm_model.save_model(f'models/saved/lstm_{args.symbol}_{args.timeframe}.pkl')
        models['lstm'] = lstm_model
    
    if 'rf' in args.models:
        logger.info("Training Random Forest model...")
        rf_model = RandomForestModel(model_type="classification")
        rf_model.set_feature_names(feature_names)
        
        # Train Random Forest
        if args.hyperparameter_tuning:
            tuning_results = rf_model.hyperparameter_tuning(
                X_train.values, y_train.values,
                method="random", n_iter=20
            )
            logger.info(f"Best RF parameters: {tuning_results['best_params']}")
        else:
            rf_model.train(X_train.values, y_train.values, X_test.values, y_test.values)
        
        # Evaluate Random Forest
        rf_metrics = rf_model.evaluate(X_test.values, y_test.values)
        logger.info(f"Random Forest Test Accuracy: {rf_metrics['accuracy']:.4f}")
        
        # Save Random Forest model
        rf_model.save_model(f'models/saved/rf_{args.symbol}_{args.timeframe}.pkl')
        models['rf'] = rf_model
        
        # Plot feature importance
        if args.plot_importance:
            rf_model.plot_feature_importance(
                top_n=20, 
                save_path=f'plots/rf_importance_{args.symbol}_{args.timeframe}.png'
            )
    
    logger.info("Model training completed!")
    return models

async def backtest_strategy(args):
    """Run backtesting on historical data."""
    logger.info("Starting backtesting...")
    
    # TODO: Implement backtesting framework
    logger.info("Backtesting framework not yet implemented")
    
async def live_trading(args):
    """Start live trading."""
    logger.info("Starting live trading...")
    
    # TODO: Implement live trading
    logger.info("Live trading not yet implemented")

def create_sample_data():
    """Create sample data for demonstration."""
    logger.info("Creating sample data...")
    
    import pandas as pd
    import numpy as np
    
    # Generate sample OHLCV data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='H')
    n_samples = len(dates)
    
    # Simulate price movement
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, n_samples)
    prices = 1.1000 + np.cumsum(returns)  # Starting at 1.1000 for EURUSD
    
    # Create OHLCV data
    data = {
        'timestamp': dates,
        'open': prices,
        'high': prices + np.random.uniform(0, 0.002, n_samples),
        'low': prices - np.random.uniform(0, 0.002, n_samples),
        'close': prices + np.random.normal(0, 0.0005, n_samples),
        'volume': np.random.randint(1000, 10000, n_samples)
    }
    
    df = pd.DataFrame(data)
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    
    # Save sample data
    Path('data/sample').mkdir(parents=True, exist_ok=True)
    df.to_csv('data/sample/EURUSD_1H_sample.csv', index=False)
    
    logger.info(f"Sample data created: {len(df)} records saved to data/sample/EURUSD_1H_sample.csv")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Forex Trading AI Agent')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'backtest', 'live', 'demo'], 
                       default='demo', help='Operation mode')
    
    # Data parameters
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Currency pair symbol')
    parser.add_argument('--timeframe', type=str, default='1H', help='Timeframe')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    
    # Model parameters
    parser.add_argument('--models', type=str, nargs='+', default=['rf'], 
                       choices=['lstm', 'rf', 'xgb', 'lgb'], help='Models to train')
    parser.add_argument('--prediction-horizon', type=int, default=24, 
                       help='Prediction horizon in time periods')
    parser.add_argument('--sequence-length', type=int, default=60, 
                       help='Sequence length for LSTM')
    parser.add_argument('--feature-selection-k', type=int, default=50, 
                       help='Number of features to select')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--hyperparameter-tuning', action='store_true', 
                       help='Perform hyperparameter tuning')
    
    # Output parameters
    parser.add_argument('--plot-importance', action='store_true', 
                       help='Plot feature importance')
    parser.add_argument('--save-plots', action='store_true', 
                       help='Save plots to files')
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path('models/saved').mkdir(parents=True, exist_ok=True)
    Path('plots').mkdir(parents=True, exist_ok=True)
    
    try:
        if args.mode == 'demo':
            logger.info("Running in demo mode...")
            create_sample_data()
            
            # Run a quick training demo with sample data
            logger.info("Running training demo with sample data...")
            args.start_date = '2020-01-01'
            args.end_date = '2022-12-31'  # Use subset for demo
            args.models = ['rf']  # Just Random Forest for demo
            args.epochs = 10  # Fewer epochs for demo
            
            await train_models(args)
            
        elif args.mode == 'train':
            await train_models(args)
            
        elif args.mode == 'backtest':
            await backtest_strategy(args)
            
        elif args.mode == 'live':
            await live_trading(args)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    # Run the main function
    asyncio.run(main())