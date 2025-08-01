"""
Yahoo Finance data provider for forex market data.
Provides historical and real-time forex data using yfinance library.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from .base_provider import BaseDataProvider

class YFinanceProvider(BaseDataProvider):
    """Yahoo Finance data provider for forex data."""
    
    def __init__(self):
        """Initialize Yahoo Finance provider."""
        super().__init__("Yahoo Finance")
        self.logger = logging.getLogger(__name__)
        
        # Yahoo Finance forex symbol mapping
        self.symbol_mapping = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'USDCHF': 'USDCHF=X',
            'NZDUSD': 'NZDUSD=X',
            'EURGBP': 'EURGBP=X',
            'EURJPY': 'EURJPY=X',
            'GBPJPY': 'GBPJPY=X'
        }
        
        # Timeframe mapping for Yahoo Finance
        self.timeframe_mapping = {
            '1H': '1h',
            '4H': '4h',
            '1D': '1d',
            '1W': '1wk',
            '1M': '1mo'
        }
    
    async def connect(self) -> bool:
        """
        Connect to Yahoo Finance (no authentication required).
        
        Returns:
            bool: Always True as no connection is needed
        """
        try:
            # Test connection by fetching a small amount of data
            test_symbol = 'EURUSD=X'
            test_data = yf.download(test_symbol, period='1d', interval='1d', progress=False)
            
            if test_data.empty:
                self.logger.error("Failed to connect to Yahoo Finance - no data returned")
                return False
            
            self.is_connected = True
            self.logger.info("Successfully connected to Yahoo Finance")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Yahoo Finance: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from Yahoo Finance.
        
        Returns:
            bool: Always True as no disconnection is needed
        """
        self.is_connected = False
        self.logger.info("Disconnected from Yahoo Finance")
        return True
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert standard forex symbol to Yahoo Finance format."""
        return self.symbol_mapping.get(symbol.upper(), f"{symbol}=X")
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to Yahoo Finance format."""
        normalized = self.normalize_timeframe(timeframe)
        return self.timeframe_mapping.get(normalized, '1d')
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Currency pair symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., '1H', '4H', '1D')
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if not self.is_connected:
            raise ConnectionError("Provider not connected")
        
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        if not self.validate_date_range(start_date, end_date):
            raise ValueError("Invalid date range")
        
        try:
            yf_symbol = self._convert_symbol(symbol)
            yf_interval = self._convert_timeframe(timeframe)
            
            self.logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")
            
            # Use asyncio to run the blocking yfinance call
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: yf.download(
                    yf_symbol,
                    start=start_date,
                    end=end_date,
                    interval=yf_interval,
                    progress=False,
                    auto_adjust=True,
                    prepost=True
                )
            )
            
            if data.empty:
                self.logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Reset index to make timestamp a column
            data = data.reset_index()
            data = data.rename(columns={'Date': 'timestamp', 'Datetime': 'timestamp'})
            
            # Ensure we have the required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in data.columns:
                    if col == 'volume':
                        data['volume'] = 0  # Forex typically doesn't have volume
                    else:
                        raise ValueError(f"Missing required column: {col}")
            
            # Apply limit if specified
            if limit and len(data) > limit:
                data = data.tail(limit)
            
            # Sort by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Retrieved {len(data)} records for {symbol}")
            return data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise
    
    async def get_real_time_data(self, symbol: str) -> Dict:
        """
        Get real-time price data from Yahoo Finance.
        
        Args:
            symbol: Currency pair symbol (e.g., 'EURUSD')
            
        Returns:
            Dict with current price information
        """
        if not self.is_connected:
            raise ConnectionError("Provider not connected")
        
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        try:
            yf_symbol = self._convert_symbol(symbol)
            
            # Get current data (last 2 days to ensure we get recent data)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2)
            
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: yf.download(
                    yf_symbol,
                    start=start_date,
                    end=end_date,
                    interval='1m',
                    progress=False
                )
            )
            
            if data.empty:
                raise ValueError(f"No real-time data available for {symbol}")
            
            # Get the latest record
            latest = data.iloc[-1]
            timestamp = data.index[-1]
            
            return {
                'symbol': symbol,
                'timestamp': timestamp,
                'bid': float(latest['Close']),  # Yahoo Finance doesn't provide bid/ask
                'ask': float(latest['Close']),
                'price': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'volume': float(latest.get('Volume', 0)),
                'provider': self.name
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time data for {symbol}: {e}")
            raise
    
    async def get_symbols(self) -> List[str]:
        """
        Get list of available forex symbols.
        
        Returns:
            List of available currency pair symbols
        """
        return list(self.symbol_mapping.keys())
    
    async def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get information about a forex symbol.
        
        Args:
            symbol: Currency pair symbol
            
        Returns:
            Dict with symbol information
        """
        if symbol not in self.symbol_mapping:
            raise ValueError(f"Symbol {symbol} not supported")
        
        # Basic forex symbol information
        symbol_info = {
            'symbol': symbol,
            'base_currency': symbol[:3],
            'quote_currency': symbol[3:6],
            'pip_size': 0.0001 if 'JPY' not in symbol else 0.01,
            'min_lot_size': 0.01,
            'max_lot_size': 100.0,
            'lot_step': 0.01,
            'margin_required': 0.02,  # 2% margin requirement
            'provider': self.name
        }
        
        return symbol_info
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a forex symbol is supported.
        
        Args:
            symbol: Currency pair symbol to validate
            
        Returns:
            bool: True if symbol is valid and supported, False otherwise
        """
        if not super().validate_symbol(symbol):
            return False
        
        return symbol.upper() in self.symbol_mapping