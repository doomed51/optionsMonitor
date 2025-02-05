import pandas as pd
import numpy as np
from typing import List
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
        # replace 0s with 1s to avoid division by 0
        volume = volume.replace(0, 1)
        # Calculate log of volume
        volume = np.log(volume)
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
        df.loc[:, 'obv'] = self.calculate_obv(df['close'], df['volume'])
        
        # Calculate MA if requested
        if ma_type != "None":
            df['obv_ma'] = self.calculate_ma(df['obv'], ma_length, ma_type)
            
            # Calculate Bollinger Bands if requested
            if use_bb:
                df['obv_bb_upper'], df['obv_bb_lower'] = self.calculate_bollinger_bands(
                    df['obv'], df['obv_ma'], ma_length, bb_mult
                )
                
        return df

    def adjusted_obv_aggregation(calls_volume, puts_volume, price_changes, theta_decay_rate):
    # Asymmetric volume weighting function
        def directional_weight(volume, price_change, alpha=0.1):
            # Higher weight for positive moves, decay-adjusted
            return volume * (1 + alpha * max(price_change, 0)) * np.exp(-theta_decay_rate)
        
            # Separate processing for calls and puts
        weighted_call_obv = sum(
            directional_weight(vol, change) 
            for vol, change in zip(calls_volume, price_changes)
        )
        
        weighted_put_obv = sum(
            directional_weight(vol, -change)  # Inverse for puts
            for vol, change in zip(puts_volume, price_changes)
        )
        
        # Net aggregated OBV with directional bias
        net_obv = weighted_call_obv - weighted_put_obv
        
        return net_obv

    def calculate_obv_across_contracts(self, contracts_hist_data: dict = {}, ma_type: str = "None", ma_length: int = 14, 
                 use_bb: bool = False, bb_mult: float = 2.0) -> pd.DataFrame:
        """
        Calculate OBV and selected moving average/bands
        
        Parameters:
        -----------
        contracts_hist_data: Dict
            Dictionary with contract and its historical data
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
            DataFrame with call OBV
        pd.DataFrame
            DataFrame with put OBV
        """
        # Validate inputs
        # required_columns = ['close', 'volume']
        # if not all(col in df.columns for col in required_columns):
        #     raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        if use_bb and ma_type == "None":
            raise ValueError("Moving average type must be specified to calculate Bollinger Bands")
        call_obv = pd.DataFrame()
        put_obv = pd.DataFrame()
        # describe contracts_hist_data

        # Calculate OBV for each option contract
        for idx in contracts_hist_data:
            df = self.calculate(contracts_hist_data[idx]['df'])
            if contracts_hist_data[idx]['contract'].right == 'C':
                call_obv = pd.concat([call_obv, df[['date', 'obv']]])
            else:
                put_obv = pd.concat([put_obv, df[['date', 'obv']]])
        
        # aggregate call and put obv
        if not call_obv.empty:
            call_obv = call_obv.groupby('date').sum().reset_index()
        if not put_obv.empty:
            put_obv = put_obv.groupby('date').sum().reset_index()
        

        # Calculate Bollinger Bands if requested
        if ma_type != "None":
            call_obv['obv_ma'] = self.calculate_ma(call_obv['obv'], ma_length, ma_type)
            put_obv['obv_ma'] = self.calculate_ma(put_obv['obv'], ma_length, ma_type)    
            if use_bb:
                call_obv['obv_bb_upper'], call_obv['obv_bb_lower'] = self.calculate_bollinger_bands(
                    call_obv['obv'], call_obv['obv_ma'], ma_length, bb_mult
                )
                put_obv['obv_bb_upper'], put_obv['obv_bb_lower'] = self.calculate_bollinger_bands(
                    put_obv['obv'], put_obv['obv_ma'], ma_length, bb_mult
                )
        return call_obv, put_obv

    def _select_nearby_contracts(self, date_options, current_price, num_strikes):
        """
        Select option contracts near the current price, considering non-uniform strike spacing
        
        Parameters:
        -----------
        date_options : pd.DataFrame
            DataFrame of options for a specific date
        current_price : float
            Current underlying price
        num_strikes : int
            Number of strikes to include on each side of the current price
        
        Returns:
        --------
        tuple: (call_contracts, put_contracts)
        """
        # Sort strikes for more efficient selection
        unique_call_strikes = sorted(date_options[date_options['right'] == 'C']['strike'].unique())
        unique_put_strikes = sorted(date_options[date_options['right'] == 'P']['strike'].unique())
        
        # Find index of strike closest to current price
        call_price_index = min(range(len(unique_call_strikes)), 
                                key=lambda i: abs(unique_call_strikes[i] - current_price))
        put_price_index = min(range(len(unique_put_strikes)), 
                            key=lambda i: abs(unique_put_strikes[i] - current_price))
        
        # Select strikes around the current price
        selected_call_strikes = unique_call_strikes[
            max(0, call_price_index - num_strikes):
            min(len(unique_call_strikes), call_price_index + num_strikes)
        ]
        
        selected_put_strikes = unique_put_strikes[
            max(0, put_price_index - num_strikes):
            min(len(unique_put_strikes), put_price_index + num_strikes)
        ]
        
        # Filter contracts based on selected strikes
        call_contracts = date_options[
            (date_options['right'] == 'C') & 
            (date_options['strike'].isin(selected_call_strikes))
        ]
        
        put_contracts = date_options[
            (date_options['right'] == 'P') & 
            (date_options['strike'].isin(selected_put_strikes))
        ]
        
        return call_contracts, put_contracts

    def add_aggregate_options_obv_to_underlying_pxhistory(self, underlying_df: pd.DataFrame, options_history_df: pd.DataFrame, num_expiries: int = 1, num_strikes: int = 5): 
        """
        Calculate OBV for options and aggregate it to underlying price history
        
        Parameters:
        -----------
        underlying_df: pd.DataFrame
            DataFrame with underlying price history
        options_history_df: pd.DataFrame
            DataFrame with options price history
        num_expiries: int
            Number of expiries to consider
        num_strikes: int
            Number of strikes to consider
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with underlying price history and aggregated OBV
        """
        # Create a copy of the underlying DataFrame to modify
        result_df = underlying_df.copy()
        # make sure date column is 19chars long 
        result_df['date'] = result_df['date'].apply(lambda x: x[:19])
        # convert to np datetime    
        result_df['date'] = pd.to_datetime(result_df['date'])

        ## drop all rows in results_df where the date is less than the earliest expiring option we have 
        result_df = result_df[result_df['date'] >= options_history_df['expiry'].min() - pd.Timedelta(days=1)]
        
        # Prepare columns for call and put aggregated OBV
        result_df['call_agg_obv'] = 0
        result_df['put_agg_obv'] = 0
        
        # Construct a dict of relevant options for each date 
        contracts_by_date = {}
        for current_date in options_history_df['date'].unique():
            # Selection optinos ohlc available for current date 
            available_option_history = options_history_df[options_history_df['date'] == current_date]

            # list of expiries that we have 
            expiries = available_option_history['expiry'].unique() 
            if len(expiries) <= num_expiries:
                continue 
            # select the expiry we want 
            target_expiry = expiries[num_expiries].date() 
            
            # Filter for relevant expiries
            target_option_history = available_option_history[available_option_history['expiry'] == np.datetime64(target_expiry)]
            contracts_by_date[current_date] = target_option_history # add current date and relevant options to dict 
        
        # calculate obv with the relevant options for each time period 
        numnotfound = 0 
        for idx, row in result_df.iterrows():
            current_date = row['date']
            current_price = row['close']
            # make sure current date is of type timestamp 
            current_date = pd.Timestamp(current_date)
            
            # Get options for this date
            if current_date not in contracts_by_date.keys():
                print(f"No options available for {current_date}, skipping to next date.")
                numnotfound += 1
                continue
            
            available_option_history = contracts_by_date[current_date]
            
            # Filter strikes within the specified range
            call_contracts, put_contracts = self._select_nearby_contracts(available_option_history, current_price, num_strikes)
            
            # Prepare dictionaries for OBV calculation
            call_contracts_hist = {}
            put_contracts_hist = {}
            for _, contract in call_contracts.iterrows():
                con_px_history = options_history_df.loc[(options_history_df['expiry'] == contract['expiry']) & (options_history_df['strike'] == contract['strike']) & (options_history_df['right'] == contract['right'])]
                call_contracts_hist[len(call_contracts_hist)] = {
                    'contract': contract,
                    'df': con_px_history
                    # 'df': contract[['date', 'close', 'volume']]
                }
            
            for _, contract in put_contracts.iterrows():
                put_px_history = options_history_df.loc[(options_history_df['expiry'] == contract['expiry']) & (options_history_df['strike'] == contract['strike']) & (options_history_df['right'] == contract['right'])]
                put_contracts_hist[len(put_contracts_hist)] = {
                    'contract': contract,
                    'df': put_px_history
                }
            
            # Calculate aggregated OBV
            if call_contracts_hist:
                call_obv, _ = self.calculate_obv_across_contracts(call_contracts_hist)
                result_df.at[idx, 'call_agg_obv'] = call_obv.loc[call_obv['date'] == current_date, 'obv'].values[0]
            
            if put_contracts_hist:
                _, put_obv = self.calculate_obv_across_contracts(put_contracts_hist)
                result_df.at[idx, 'put_agg_obv'] = put_obv.loc[put_obv['date'] == current_date, 'obv'].values[0]
        if numnotfound > 0:
            print(f"Number of dates with no options available: {numnotfound}")
        return result_df

# Example usage:
if __name__ == "__main__":
    # Create sample data
    # df = pd.DataFrame({
    #     'close': [10, 11, 10.5, 11.5, 12, 11.8, 11.5],
    #     'volume': [100, 150, 120, 200, 180, 160, 140]
    # })
    
    # # Initialize and calculate OBV
    # obv = OnBalanceVolume()
    
    # # Example 1: Calculate with EMA and Bollinger Bands
    # result1 = obv.calculate(
    #     df,
    #     ma_type="EMA",
    #     ma_length=3,
    #     use_bb=True,
    #     bb_mult=2.0
    # )
    # print("\nEMA with Bollinger Bands:")
    # print(result1)
    
    # # Example 2: Calculate with WMA and Bollinger Bands
    # result2 = obv.calculate(
    #     df,
    #     ma_type="WMA",
    #     ma_length=3,
    #     use_bb=True,
    #     bb_mult=2.0
    # )
    # print("\nWMA with Bollinger Bands:")
    # print(result2)
    import duckdb 
    import pandas as pd 
    import matplotlib.pyplot as plt
    import sqlite3

    options_db_path = 'F:\workbench\optionsDataManager\data\options_data.db'
    underlying_db_path = 'F:\workbench\historicalData\\venv\saveHistoricalData\data\historicalData_index.db'

    conn_options = duckdb.connect(options_db_path)
    conn_underlying = sqlite3.connect(underlying_db_path)

    read_query = 'SELECT * FROM options_historical_data'
    options_historical_data_df = conn_options.execute(read_query).fetchdf()

    read_query = 'SELECT * FROM SPY_stock_1min'
    underlying_historical_data_df = pd.read_sql_query(read_query, conn_underlying).head(100) 

    obv = OnBalanceVolume()
    # print unique expiries in options hist data 
    result_df = obv.add_aggregate_options_obv_to_underlying_pxhistory(underlying_historical_data_df, options_historical_data_df, num_expiries=1, num_strikes=5)

    print(result_df.tail(30))
