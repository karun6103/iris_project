"""
Base data provider class for forex market data.
Defines the interface that all data providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

class BaseDataProvider(ABC):
    """Abstract base class for all data providers."""
    
    def __init__(self, name: str):
        """
        Initialize the data provider.
        
        Args:
            name: Name of the data provider
        """
        self.name = name
        self.is_connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the data provider.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the data provider.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: Currency pair symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., '1H', '4H', '1D')
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    async def get_real_time_data(self, symbol: str) -> Dict:
        """
        Get real-time price data for a symbol.
        
        Args:
            symbol: Currency pair symbol (e.g., 'EURUSD')
            
        Returns:
            Dict with current price information
        """
        pass
    
    @abstractmethod
    async def get_symbols(self) -> List[str]:
        """
        Get list of available symbols.
        
        Returns:
            List of available currency pair symbols
        """
        pass
    
    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get information about a specific symbol.
        
        Args:
            symbol: Currency pair symbol
            
        Returns:
            Dict with symbol information (spread, pip_size, etc.)
        """
        pass
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is supported.
        
        Args:
            symbol: Currency pair symbol to validate
            
        Returns:
            bool: True if symbol is valid, False otherwise
        """
        # Basic validation - can be overridden by specific providers
        if not symbol or len(symbol) < 6:
            return False
        return True
    
    def normalize_timeframe(self, timeframe: str) -> str:
        """
        Normalize timeframe string to standard format.
        
        Args:
            timeframe: Timeframe string (e.g., '1h', '4H', '1D')
            
        Returns:
            Normalized timeframe string
        """
        timeframe = timeframe.upper()
        # Map common variations to standard format
        mapping = {
            '1H': '1H', '1HOUR': '1H', '60M': '1H',
            '4H': '4H', '4HOUR': '4H', '240M': '4H',
            '1D': '1D', '1DAY': '1D', '24H': '1D',
            '1W': '1W', '1WEEK': '1W',
            '1M': '1M', '1MONTH': '1M'
        }
        return mapping.get(timeframe, timeframe)
    
    def validate_date_range(self, start_date: datetime, end_date: datetime) -> bool:
        """
        Validate date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            bool: True if date range is valid, False otherwise
        """
        if start_date >= end_date:
            return False
        if end_date > datetime.now():
            return False
        return True
    
    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.name} Provider (Connected: {self.is_connected})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the provider."""
        return f"<{self.__class__.__name__}(name='{self.name}', connected={self.is_connected})>"