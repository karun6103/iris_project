"""
Data preprocessor for forex trading AI.
Handles data fetching, cleaning, feature engineering, and ML preparation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from .providers.base_provider import BaseDataProvider
from .providers.yfinance_provider import YFinanceProvider
from utils.indicators import TechnicalIndicators
from utils.config import get_config

class DataPreprocessor:
    """Comprehensive data preprocessor for forex trading."""
    
    def __init__(self, provider: Optional[BaseDataProvider] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            provider: Data provider instance. If None, uses YFinanceProvider.
        """
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        
        # Initialize data provider
        self.provider = provider or YFinanceProvider()
        
        # Initialize technical indicators calculator
        self.indicators = TechnicalIndicators()
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        # Feature selection
        self.feature_selector = None
        self.selected_features = None
        
        # Cache for processed data
        self.data_cache = {}
    
    async def fetch_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical data from the data provider.
        
        Args:
            symbol: Currency pair symbol
            timeframe: Timeframe for data
            start_date: Start date
            end_date: End date
            limit: Maximum number of records
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}_{limit}"
        
        if cache_key in self.data_cache:
            self.logger.info(f"Using cached data for {symbol}")
            return self.data_cache[cache_key].copy()
        
        if not self.provider.is_connected:
            await self.provider.connect()
        
        data = await self.provider.get_historical_data(
            symbol, timeframe, start_date, end_date, limit
        )
        
        if not data.empty:
            self.data_cache[cache_key] = data.copy()
        
        return data
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values, outliers, and duplicates.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Cleaning data...")
        
        # Make a copy to avoid modifying original data
        cleaned_df = df.copy()
        
        # Remove duplicates
        initial_length = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=['timestamp'])
        if len(cleaned_df) < initial_length:
            self.logger.info(f"Removed {initial_length - len(cleaned_df)} duplicate records")
        
        # Sort by timestamp
        cleaned_df = cleaned_df.sort_values('timestamp').reset_index(drop=True)
        
        # Handle missing values
        missing_count = cleaned_df.isnull().sum().sum()
        if missing_count > 0:
            self.logger.warning(f"Found {missing_count} missing values")
            
            # Forward fill missing values
            cleaned_df = cleaned_df.fillna(method='ffill')
            
            # If still missing values at the beginning, backward fill
            cleaned_df = cleaned_df.fillna(method='bfill')
            
            # Drop any remaining rows with missing values
            cleaned_df = cleaned_df.dropna()
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (cleaned_df['high'] < cleaned_df['low']) |
            (cleaned_df['high'] < cleaned_df['open']) |
            (cleaned_df['high'] < cleaned_df['close']) |
            (cleaned_df['low'] > cleaned_df['open']) |
            (cleaned_df['low'] > cleaned_df['close'])
        )
        
        if invalid_ohlc.any():
            self.logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC relationships")
            cleaned_df = cleaned_df[~invalid_ohlc]
        
        # Remove extreme outliers using IQR method
        for col in ['open', 'high', 'low', 'close']:
            Q1 = cleaned_df[col].quantile(0.01)
            Q3 = cleaned_df[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
            if outliers.any():
                self.logger.warning(f"Removing {outliers.sum()} outliers in {col}")
                cleaned_df = cleaned_df[~outliers]
        
        # Reset index
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        self.logger.info(f"Data cleaning completed. Final shape: {cleaned_df.shape}")
        return cleaned_df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        self.logger.info("Adding technical indicators...")
        return self.indicators.calculate_all_indicators(df)
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features like returns, volatility, etc.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional price features
        """
        result = df.copy()
        
        # Returns
        for period in [1, 3, 5, 10, 20]:
            result[f'return_{period}'] = result['close'].pct_change(period)
            result[f'log_return_{period}'] = np.log(result['close'] / result['close'].shift(period))
        
        # Volatility (rolling standard deviation of returns)
        for period in [5, 10, 20]:
            result[f'volatility_{period}'] = result['return_1'].rolling(window=period).std()
        
        # Price ratios
        result['hl_ratio'] = result['high'] / result['low']
        result['oc_ratio'] = result['open'] / result['close']
        
        # Gap analysis
        result['gap'] = (result['open'] - result['close'].shift(1)) / result['close'].shift(1)
        
        # Intraday range
        result['intraday_range'] = (result['high'] - result['low']) / result['open']
        
        # Body and shadow ratios (candlestick analysis)
        result['body_size'] = abs(result['close'] - result['open']) / result['open']
        result['upper_shadow'] = (result['high'] - np.maximum(result['open'], result['close'])) / result['open']
        result['lower_shadow'] = (np.minimum(result['open'], result['close']) - result['low']) / result['open']
        
        # Price position within the range
        result['price_position'] = (result['close'] - result['low']) / (result['high'] - result['low'])
        
        return result
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with time features
        """
        result = df.copy()
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(result['timestamp']):
            result['timestamp'] = pd.to_datetime(result['timestamp'])
        
        # Extract time components
        result['hour'] = result['timestamp'].dt.hour
        result['day_of_week'] = result['timestamp'].dt.dayofweek
        result['day_of_month'] = result['timestamp'].dt.day
        result['month'] = result['timestamp'].dt.month
        result['quarter'] = result['timestamp'].dt.quarter
        
        # Cyclical encoding for time features
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        result['day_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        # Market session indicators (assuming UTC timestamps)
        result['asian_session'] = ((result['hour'] >= 0) & (result['hour'] < 8)).astype(int)
        result['european_session'] = ((result['hour'] >= 8) & (result['hour'] < 16)).astype(int)
        result['american_session'] = ((result['hour'] >= 16) & (result['hour'] < 24)).astype(int)
        
        # Weekend indicator
        result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
        
        return result
    
    def add_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Add lagged features for specified columns.
        
        Args:
            df: DataFrame
            columns: List of column names to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        result = df.copy()
        
        for col in columns:
            if col in result.columns:
                for lag in lags:
                    result[f'{col}_lag_{lag}'] = result[col].shift(lag)
        
        return result
    
    def add_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """
        Add rolling statistical features.
        
        Args:
            df: DataFrame
            columns: List of column names to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        result = df.copy()
        
        for col in columns:
            if col in result.columns:
                for window in windows:
                    result[f'{col}_mean_{window}'] = result[col].rolling(window=window).mean()
                    result[f'{col}_std_{window}'] = result[col].rolling(window=window).std()
                    result[f'{col}_min_{window}'] = result[col].rolling(window=window).min()
                    result[f'{col}_max_{window}'] = result[col].rolling(window=window).max()
                    result[f'{col}_skew_{window}'] = result[col].rolling(window=window).skew()
        
        return result
    
    def create_target_variables(self, df: pd.DataFrame, horizons: List[int] = None) -> pd.DataFrame:
        """
        Create target variables for prediction.
        
        Args:
            df: DataFrame with price data
            horizons: List of prediction horizons
            
        Returns:
            DataFrame with target variables
        """
        if horizons is None:
            horizons = [1, 3, 5, 10, 24]  # Default horizons
        
        result = df.copy()
        
        for horizon in horizons:
            # Price direction (binary classification)
            future_price = result['close'].shift(-horizon)
            result[f'direction_{horizon}'] = (future_price > result['close']).astype(int)
            
            # Price change (regression)
            result[f'price_change_{horizon}'] = (future_price - result['close']) / result['close']
            
            # Log return (regression)
            result[f'log_return_target_{horizon}'] = np.log(future_price / result['close'])
            
            # High/Low targets for the next N periods
            future_high = result['high'].rolling(window=horizon).max().shift(-horizon)
            future_low = result['low'].rolling(window=horizon).min().shift(-horizon)
            
            result[f'future_high_{horizon}'] = (future_high - result['close']) / result['close']
            result[f'future_low_{horizon}'] = (result['close'] - future_low) / result['close']
        
        return result
    
    def select_features(self, df: pd.DataFrame, target_column: str, k: int = 50, method: str = 'f_regression') -> pd.DataFrame:
        """
        Select the most important features using statistical methods.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of the target column
            k: Number of features to select
            method: Feature selection method ('f_regression' or 'mutual_info')
            
        Returns:
            DataFrame with selected features
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in ['timestamp', target_column]]
        X = df[feature_columns].select_dtypes(include=[np.number])
        y = df[target_column]
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            self.logger.warning("No valid samples for feature selection")
            return df
        
        # Select scoring function
        if method == 'f_regression':
            score_func = f_regression
        elif method == 'mutual_info':
            score_func = mutual_info_regression
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Perform feature selection
        self.feature_selector = SelectKBest(score_func=score_func, k=min(k, X_clean.shape[1]))
        X_selected = self.feature_selector.fit_transform(X_clean, y_clean)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        self.logger.info(f"Selected {len(self.selected_features)} features using {method}")
        
        # Return DataFrame with selected features
        result_columns = ['timestamp'] + self.selected_features + [target_column]
        return df[result_columns]
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: DataFrame to scale
            method: Scaling method ('standard', 'minmax', 'robust')
            exclude_columns: Columns to exclude from scaling
            
        Returns:
            Scaled DataFrame
        """
        if exclude_columns is None:
            exclude_columns = ['timestamp']
        
        result = df.copy()
        
        # Select numerical columns to scale
        numerical_columns = result.select_dtypes(include=[np.number]).columns
        columns_to_scale = [col for col in numerical_columns if col not in exclude_columns]
        
        if not columns_to_scale:
            return result
        
        # Get the appropriate scaler
        if method not in self.scalers:
            raise ValueError(f"Unknown scaling method: {method}")
        
        scaler = self.scalers[method]
        
        # Fit and transform the data
        result[columns_to_scale] = scaler.fit_transform(result[columns_to_scale])
        
        self.logger.info(f"Scaled {len(columns_to_scale)} features using {method} scaling")
        
        return result
    
    async def process_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        target_horizons: List[int] = None,
        feature_selection_k: int = 50,
        scaling_method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Complete data processing pipeline.
        
        Args:
            symbol: Currency pair symbol
            timeframe: Timeframe for data
            start_date: Start date
            end_date: End date
            target_horizons: List of prediction horizons for target variables
            feature_selection_k: Number of features to select
            scaling_method: Method for feature scaling
            
        Returns:
            Fully processed DataFrame ready for ML
        """
        self.logger.info(f"Starting data processing for {symbol} {timeframe}")
        
        # 1. Fetch raw data
        raw_data = await self.fetch_data(symbol, timeframe, start_date, end_date)
        
        if raw_data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # 2. Clean data
        cleaned_data = self.clean_data(raw_data)
        
        # 3. Add technical indicators
        data_with_indicators = self.add_technical_indicators(cleaned_data)
        
        # 4. Add price features
        data_with_price_features = self.add_price_features(data_with_indicators)
        
        # 5. Add time features
        data_with_time_features = self.add_time_features(data_with_price_features)
        
        # 6. Add lag features
        lag_columns = ['close', 'return_1', 'rsi_14', 'macd']
        data_with_lags = self.add_lag_features(data_with_time_features, lag_columns, [1, 2, 3, 5])
        
        # 7. Add rolling features
        rolling_columns = ['close', 'return_1', 'volatility_5']
        data_with_rolling = self.add_rolling_features(data_with_lags, rolling_columns, [5, 10, 20])
        
        # 8. Create target variables
        if target_horizons is None:
            target_horizons = [self.config.ml.prediction_horizon]
        
        data_with_targets = self.create_target_variables(data_with_rolling, target_horizons)
        
        # 9. Feature selection (if enabled)
        if self.config.ml.feature_selection and feature_selection_k > 0:
            primary_target = f'direction_{target_horizons[0]}'
            if primary_target in data_with_targets.columns:
                data_selected = self.select_features(data_with_targets, primary_target, feature_selection_k)
            else:
                data_selected = data_with_targets
        else:
            data_selected = data_with_targets
        
        # 10. Scale features
        final_data = self.scale_features(
            data_selected, 
            method=scaling_method,
            exclude_columns=['timestamp'] + [col for col in data_selected.columns if col.startswith('direction_') or col.startswith('price_change_')]
        )
        
        # 11. Remove rows with missing values (from indicators and lag features)
        final_data = final_data.dropna()
        
        self.logger.info(f"Data processing completed. Final shape: {final_data.shape}")
        return final_data
    
    def prepare_ml_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for machine learning by splitting into train/test sets.
        
        Args:
            df: Processed DataFrame
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Remove timestamp and target columns from features
        feature_columns = [col for col in df.columns if col not in ['timestamp', target_column]]
        X = df[feature_columns]
        y = df[target_column]
        
        # Time-based split (important for time series data)
        split_index = int(len(df) * (1 - test_size))
        
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        self.logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test