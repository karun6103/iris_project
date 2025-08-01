"""
Technical indicators library for forex trading.
Provides a comprehensive set of technical indicators using TA-Lib and pandas.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Using pandas-based implementations.")

class TechnicalIndicators:
    """Technical indicators calculator for forex data."""
    
    def __init__(self):
        """Initialize the technical indicators calculator."""
        self.logger = logging.getLogger(__name__)
    
    def sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.SMA(data.values, timeperiod=period), index=data.index)
        return data.rolling(window=period).mean()
    
    def ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.EMA(data.values, timeperiod=period), index=data.index)
        return data.ewm(span=period).mean()
    
    def rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.RSI(data.values, timeperiod=period), index=data.index)
        
        # Pandas implementation
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        if TALIB_AVAILABLE:
            macd_line, macd_signal, macd_hist = talib.MACD(
                data.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            return {
                'macd': pd.Series(macd_line, index=data.index),
                'signal': pd.Series(macd_signal, index=data.index),
                'histogram': pd.Series(macd_hist, index=data.index)
            }
        
        # Pandas implementation
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        macd_line = ema_fast - ema_slow
        macd_signal = self.ema(macd_line, signal)
        macd_hist = macd_line - macd_signal
        
        return {
            'macd': macd_line,
            'signal': macd_signal,
            'histogram': macd_hist
        }
    
    def bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands."""
        if TALIB_AVAILABLE:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                data.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
            return {
                'upper': pd.Series(bb_upper, index=data.index),
                'middle': pd.Series(bb_middle, index=data.index),
                'lower': pd.Series(bb_lower, index=data.index)
            }
        
        # Pandas implementation
        sma = self.sma(data, period)
        std = data.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator."""
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(
                high.values, low.values, close.values,
                fastk_period=k_period, slowk_period=d_period, slowd_period=d_period
            )
            return {
                'k': pd.Series(slowk, index=close.index),
                'd': pd.Series(slowd, index=close.index)
            }
        
        # Pandas implementation
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), 
                           index=close.index)
        
        # Pandas implementation
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period),
                           index=close.index)
        
        # Simplified pandas implementation
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        atr_val = self.atr(high, low, close, period)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_val)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_val)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean()
    
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.CCI(high.values, low.values, close.values, timeperiod=period),
                           index=close.index)
        
        # Pandas implementation
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (tp - sma_tp) / (0.015 * mad)
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.WILLR(high.values, low.values, close.values, timeperiod=period),
                           index=close.index)
        
        # Pandas implementation
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    def momentum(self, data: pd.Series, period: int = 10) -> pd.Series:
        """Momentum indicator."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.MOM(data.values, timeperiod=period), index=data.index)
        return data.diff(period)
    
    def roc(self, data: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.ROC(data.values, timeperiod=period), index=data.index)
        return ((data / data.shift(period)) - 1) * 100
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.OBV(close.values, volume.values), index=close.index)
        
        # Pandas implementation
        direction = np.where(close.diff() > 0, 1, np.where(close.diff() < 0, -1, 0))
        return (direction * volume).cumsum()
    
    def ichimoku(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Ichimoku Cloud components."""
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    def fibonacci_retracement(self, high: pd.Series, low: pd.Series) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        max_price = high.max()
        min_price = low.min()
        diff = max_price - min_price
        
        levels = {
            'level_0': max_price,
            'level_236': max_price - 0.236 * diff,
            'level_382': max_price - 0.382 * diff,
            'level_500': max_price - 0.500 * diff,
            'level_618': max_price - 0.618 * diff,
            'level_786': max_price - 0.786 * diff,
            'level_100': min_price
        }
        
        return levels
    
    def pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate pivot points and support/resistance levels."""
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        result = df.copy()
        
        try:
            # Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                result[f'sma_{period}'] = self.sma(df['close'], period)
                result[f'ema_{period}'] = self.ema(df['close'], period)
            
            # Oscillators
            result['rsi_14'] = self.rsi(df['close'])
            result['rsi_21'] = self.rsi(df['close'], 21)
            
            # MACD
            macd_data = self.macd(df['close'])
            result['macd'] = macd_data['macd']
            result['macd_signal'] = macd_data['signal']
            result['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = self.bollinger_bands(df['close'])
            result['bb_upper'] = bb_data['upper']
            result['bb_middle'] = bb_data['middle']
            result['bb_lower'] = bb_data['lower']
            result['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
            result['bb_position'] = (df['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
            
            # Stochastic
            stoch_data = self.stochastic(df['high'], df['low'], df['close'])
            result['stoch_k'] = stoch_data['k']
            result['stoch_d'] = stoch_data['d']
            
            # Other indicators
            result['atr_14'] = self.atr(df['high'], df['low'], df['close'])
            result['adx_14'] = self.adx(df['high'], df['low'], df['close'])
            result['cci_20'] = self.cci(df['high'], df['low'], df['close'])
            result['williams_r'] = self.williams_r(df['high'], df['low'], df['close'])
            result['momentum_10'] = self.momentum(df['close'])
            result['roc_10'] = self.roc(df['close'])
            
            # Volume indicators (if volume data is available)
            if 'volume' in df.columns and df['volume'].sum() > 0:
                result['obv'] = self.obv(df['close'], df['volume'])
            
            # Price action indicators
            result['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
            result['close_open_pct'] = (df['close'] - df['open']) / df['open'] * 100
            
            # Ichimoku
            ichimoku_data = self.ichimoku(df['high'], df['low'], df['close'])
            for key, value in ichimoku_data.items():
                result[f'ichimoku_{key}'] = value
            
            # Pivot points (using previous period's data)
            pivot_data = self.pivot_points(df['high'].shift(1), df['low'].shift(1), df['close'].shift(1))
            for key, value in pivot_data.items():
                result[f'pivot_{key}'] = value
            
            self.logger.info(f"Calculated {len(result.columns) - len(df.columns)} technical indicators")
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            raise
        
        return result