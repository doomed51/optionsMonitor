import pandas as pd
import numpy as np

class OnBalanceVolume:
    def __init__(self):
        """Initialize the On Balance Volume (OBV) indicator"""
        self.ma_types = ["None", "SMA", "EMA", "SMMA (RMA)", "WMA", "VWMA"]
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate basic On Balance Volume
        
        Parameters:
        -----------
        close: pd.Series
            Series of closing prices
        volume: pd.Series
            Series of volume values
            
        Returns:
        --------
        pd.Series
            On Balance Volume values
        """
        if volume.sum() == 0:
            raise RuntimeError("No volume provided.")
            
        # Calculate price change direction
        price_change = close.diff()
        sign = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        
        # Calculate OBV
        obv = (sign * volume).cumsum()
        return obv
    
    def calculate_ma(self, source: pd.Series, length: int, ma_type: str) -> pd.Series:
        """
        Calculate various types of moving averages
        
        Parameters:
        -----------
        source: pd.Series
            Data to calculate MA on
        length: int
            Moving average period
        ma_type: str
            Type of moving average to calculate
            
        Returns:
        --------
        pd.Series
            Moving average values
        """
        if ma_type not in self.ma_types:
            raise ValueError(f"MA type must be one of {self.ma_types}")
            
        if ma_type == "None":
            return pd.Series(index=source.index, dtype=float)
        
        if ma_type == "SMA":
            return source.rolling(window=length).mean()
        elif ma_type == "EMA":
            return source.ewm(span=length, adjust=False).mean()
        elif ma_type == "SMMA (RMA)":
            return source.ewm(alpha=1/length, adjust=False).mean()
        elif ma_type == "WMA":
            weights = np.arange(1, length + 1)
            return source.rolling(window=length).apply(
                lambda x: np.sum(weights * x) / weights.sum()
            )
        elif ma_type == "VWMA":
            # Note: This is a simplified VWMA as we don't have volume in this context
            return source.rolling(window=length).mean()
            
    def calculate_bollinger_bands(self, source: pd.Series, ma: pd.Series, length: int, 
                                mult: float) -> tuple[pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands based on any moving average
        
        Parameters:
        -----------
        source: pd.Series
            Original data series
        ma: pd.Series
            Moving average series
        length: int
            Period for standard deviation calculation
        mult: float
            Standard deviation multiplier
            
        Returns:
        --------
        tuple[pd.Series, pd.Series]
            Upper and lower Bollinger Bands
        """
        std = source.rolling(window=length).std()
        upper_band = ma + (std * mult)
        lower_band = ma - (std * mult)
        return upper_band, lower_band
            
    def calculate(self, df: pd.DataFrame, ma_type: str = "None", ma_length: int = 14, 
                 use_bb: bool = False, bb_mult: float = 2.0) -> pd.DataFrame:
        """
        Calculate OBV and selected moving average/bands
        
        Parameters:
        -----------
        df: pd.DataFrame
            DataFrame with 'close' and 'volume' columns
        ma_type: str
            Type of moving average to calculate
        ma_length: int
            Moving average period
        use_bb: bool
            Whether to calculate Bollinger Bands
        bb_mult: float
            Bollinger Bands standard deviation multiplier
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with OBV and selected indicators
        """
        # Validate inputs
        required_columns = ['close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        if use_bb and ma_type == "None":
            raise ValueError("Moving average type must be specified to calculate Bollinger Bands")
            
        # Calculate OBV
        df['obv'] = self.calculate_obv(df['close'], df['volume'])
        
        # Calculate MA if requested
        if ma_type != "None":
            df['obv_ma'] = self.calculate_ma(df['obv'], ma_length, ma_type)
            
            # Calculate Bollinger Bands if requested
            if use_bb:
                df['obv_bb_upper'], df['obv_bb_lower'] = self.calculate_bollinger_bands(
                    df['obv'], df['obv_ma'], ma_length, bb_mult
                )
                
        return df

# Example usage:
if __name__ == "__main__":
    # Create sample data
    df = pd.DataFrame({
        'close': [10, 11, 10.5, 11.5, 12, 11.8, 11.5],
        'volume': [100, 150, 120, 200, 180, 160, 140]
    })
    
    # Initialize and calculate OBV
    obv = OnBalanceVolume()
    
    # Example 1: Calculate with EMA and Bollinger Bands
    result1 = obv.calculate(
        df,
        ma_type="EMA",
        ma_length=3,
        use_bb=True,
        bb_mult=2.0
    )
    print("\nEMA with Bollinger Bands:")
    print(result1)
    
    # Example 2: Calculate with WMA and Bollinger Bands
    result2 = obv.calculate(
        df,
        ma_type="WMA",
        ma_length=3,
        use_bb=True,
        bb_mult=2.0
    )
    print("\nWMA with Bollinger Bands:")
    print(result2)