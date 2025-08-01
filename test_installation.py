#!/usr/bin/env python3
"""
Test script to verify the installation and basic functionality
of the Forex Trading AI Agent.
"""

import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        # Test basic imports
        import numpy as np
        import pandas as pd
        import sklearn
        logger.info("✓ Basic scientific libraries imported successfully")
        
        # Test our modules
        from utils.config import get_config
        from utils.indicators import TechnicalIndicators
        from data.preprocessor import DataPreprocessor
        from data.providers.yfinance_provider import YFinanceProvider
        from models.rf_model import RandomForestModel
        logger.info("✓ Custom modules imported successfully")
        
        # Test TensorFlow (optional)
        try:
            import tensorflow as tf
            logger.info("✓ TensorFlow imported successfully")
            from models.lstm_model import LSTMModel
            logger.info("✓ LSTM model imported successfully")
        except ImportError:
            logger.warning("⚠ TensorFlow not available - LSTM model will not work")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    try:
        from utils.config import get_config
        config = get_config()
        
        logger.info(f"✓ Configuration loaded successfully")
        logger.info(f"  - Currency pairs: {config.trading.currency_pairs}")
        logger.info(f"  - Timeframes: {config.trading.timeframes}")
        logger.info(f"  - Initial balance: ${config.trading.initial_balance}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration error: {e}")
        return False

def test_technical_indicators():
    """Test technical indicators calculation."""
    logger.info("Testing technical indicators...")
    
    try:
        import pandas as pd
        import numpy as np
        from utils.indicators import TechnicalIndicators
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        np.random.seed(42)
        prices = 1.1000 + np.cumsum(np.random.normal(0, 0.001, 100))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 0.002, 100),
            'low': prices - np.random.uniform(0, 0.002, 100),
            'close': prices + np.random.normal(0, 0.0005, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Fix OHLC relationships
        df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
        df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
        
        # Calculate indicators
        indicators = TechnicalIndicators()
        result = indicators.calculate_all_indicators(df)
        
        logger.info(f"✓ Technical indicators calculated successfully")
        logger.info(f"  - Original columns: {len(df.columns)}")
        logger.info(f"  - With indicators: {len(result.columns)}")
        logger.info(f"  - Added indicators: {len(result.columns) - len(df.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Technical indicators error: {e}")
        return False

def test_data_provider():
    """Test data provider functionality."""
    logger.info("Testing data provider...")
    
    try:
        from data.providers.yfinance_provider import YFinanceProvider
        
        provider = YFinanceProvider()
        symbols = provider.get_symbols()
        
        logger.info(f"✓ Data provider initialized successfully")
        logger.info(f"  - Available symbols: {len(symbols)}")
        logger.info(f"  - Sample symbols: {symbols[:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data provider error: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    logger.info("Testing model creation...")
    
    try:
        from models.rf_model import RandomForestModel
        
        # Test Random Forest
        rf_model = RandomForestModel(model_type="classification")
        logger.info("✓ Random Forest model created successfully")
        
        # Test LSTM if TensorFlow is available
        try:
            from models.lstm_model import LSTMModel
            lstm_model = LSTMModel(sequence_length=60, model_type="classification")
            logger.info("✓ LSTM model created successfully")
        except ImportError:
            logger.warning("⚠ LSTM model not available (TensorFlow required)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model creation error: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist or can be created."""
    logger.info("Testing directory structure...")
    
    required_dirs = [
        'logs',
        'models/saved',
        'plots',
        'data/sample'
    ]
    
    try:
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Directory created/verified: {dir_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Directory creation error: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("=" * 50)
    logger.info("Forex Trading AI Agent - Installation Test")
    logger.info("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Technical Indicators", test_technical_indicators),
        ("Data Provider", test_data_provider),
        ("Model Creation", test_model_creation),
        ("Directory Structure", test_directory_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
        logger.info(f"--- {test_name} Complete ---")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The system is ready to use.")
        logger.info("\nTo get started, try:")
        logger.info("  python main.py --mode demo")
    else:
        logger.warning(f"⚠ {total - passed} tests failed. Please check the errors above.")
        logger.info("\nCommon issues:")
        logger.info("  - Missing dependencies: pip install -r requirements.txt")
        logger.info("  - TA-Lib installation: see README.md for platform-specific instructions")
        logger.info("  - TensorFlow for LSTM: pip install tensorflow")
    
    logger.info("=" * 50)
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)