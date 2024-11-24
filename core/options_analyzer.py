import matplotlib.pyplot as plt
import queue
from matplotlib.animation import FuncAnimation
from ib_insync import *
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime
from indicators.onBalanceVolume import OnBalanceVolume

from core.contract_manager import OptionsContractManager
from core.options_data_retriever import OptionsDataRetriever

# from indicators import onBalanceVolume

class OptionsAnalyzer:
    def __init__(self):
        self.ib = IB()
        self.obv = OnBalanceVolume()
        # self.obv_data = pd.DataFrame()
        # obv_data is a tuple of (ticker, historicaldata_df)
        self.obv_data = []
        
        # Set up real-time plot
        plt.style.use('dark_background')
        plt.ion()
        self.fig, self.ax = plt.subplots(2, figsize=(11,7), sharex=True)
        self.line, = self.ax[0].plot([], [], label='Aggregate OBV')
        self.ax[0].set_title('Real-Time Options Aggregate On-Balance Volume')
        self.ax[0].set_xlabel('Time')
        self.ax[0].set_ylabel('OBV')
        self.ax[0].legend()
        
    def connect(self, host='127.0.0.1', port=7496, clientId=10):
        """Connect to TWS/IBGateway"""
        self.ib.connect(host, port, clientId)
                
    def get_historical_data(self, 
                          contracts: List[Contract], 
                          duration: str = '1 D',
                          bar_size: str = '1 min') -> pd.DataFrame:
        """Get historical data for list of contracts"""
        dfs = []
        for contract in contracts:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if bars:
                df = util.df(bars)
                df['contract'] = contract.localSymbol
                dfs.append(df)
                
        return pd.concat(dfs) if dfs else pd.DataFrame()
        
    def calculate_aggregate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate OBV across all options in dataframe"""
        # Group by timestamp to aggregate across all options
        agg_df = df.groupby('date').agg({
            'close': 'mean',
            'volume': 'sum'
        }).reset_index()
        obv = onBalanceVolume.OnBalanceVolume()
        # Calculate aggregate OBV
        agg_df = obv.calculate(agg_df, ma_type='SMA', ma_length=45, use_bb=True, bb_mult=1.0)
        return agg_df
        
    def calculate_obv(self, close: float, volume: float) -> float:
        """Simple OBV calculation"""
        if not hasattr(self, '_prev_close'):
            self._prev_close = close
            self._prev_obv = 0
            return 0
        
        if close > self._prev_close:
            obv = self._prev_obv + volume
        elif close < self._prev_close:
            obv = self._prev_obv - volume
        else:
            obv = self._prev_obv
        
        self._prev_close = close
        self._prev_obv = obv
        return obv

    def monitor_realtime(self, symbol: str, num_otm_strikes: int = 5, num_days_to_expiry: int = 1, num_periods_to_plot: int = 360):
            """Monitor options in realtime with live plotting"""

            def update_plot(frame):
                """Update the plot for real-time animation."""
                print('Updating plot...')

                contract_manager = OptionsContractManager(self.ib)
                data_retriever = OptionsDataRetriever(self.ib)

                # get otm contracts 
                otm_cons = contract_manager.get_otm_options(symbol, num_otm_strikes, num_days_to_expiry)
                all_contracts = otm_cons['calls'] + otm_cons['puts']

                # get history for otm contracts 
                contracts_and_hist_data = data_retriever.get_historical_data(all_contracts, duration='1 D')

                # Aggregate obv for calls and puts 
                call_obv, put_obv = self.obv.calculate_obv_across_contracts(contracts_and_hist_data, ma_type='SMA', ma_length=20, use_bb=True, bb_mult=0.5)

                ## add ratio of callobv to putobv
                call_obv['obv_call-put-ratio'] = call_obv['obv'] - put_obv['obv']
                call_obv['obv_cp_ratio_ma'] = self.obv.calculate_ma(call_obv['obv_call-put-ratio'], ma_type='EMA', length=40)
                call_obv['obv_cp_ratio_upper'], call_obv['obv_cp_ratio_lower']= self.obv.calculate_bollinger_bands(call_obv['obv_call-put-ratio'], call_obv['obv_cp_ratio_ma'], length=40, mult=0.5)

                # convert date column to string for better plots 
                call_obv['date'] = call_obv['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                put_obv['date'] = put_obv['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

                # plot stats                      
                self.ax[0].clear()
                    
                self.ax[0].plot(call_obv['date'].tail(num_periods_to_plot), call_obv['obv'].tail(num_periods_to_plot), label='Call OBV', color='green')
                self.ax[0].plot(call_obv['date'].tail(num_periods_to_plot), call_obv['obv_ma'].tail(num_periods_to_plot), label='Call OBV MA')
                self.ax[0].plot(call_obv['date'].tail(num_periods_to_plot), call_obv['obv_bb_upper'].tail(num_periods_to_plot), label='Call OBV Upper Band')
                self.ax[0].plot(call_obv['date'].tail(num_periods_to_plot), call_obv['obv_bb_lower'].tail(num_periods_to_plot), label='Call OBV Lower Band')
                # self.ax[0].plot(call_obv['date'].tail(num_periods_to_plot), call_obv['obv_upper_band2'].tail(num_periods_to_plot), label='Call OBV Upper Band 2')
                # self.ax[0].plot(call_obv['date'].tail(num_periods_to_plot), call_obv['obv_lower_band2'].tail(num_periods_to_plot), label='Call OBV Lower Band 2')

                self.ax[0].plot(put_obv['date'].tail(num_periods_to_plot), put_obv['obv'].tail(num_periods_to_plot), label='Put OBV', color='red')
                self.ax[0].plot(put_obv['date'].tail(num_periods_to_plot), put_obv['obv_ma'].tail(num_periods_to_plot), label='Put OBV MA')
                self.ax[0].plot(put_obv['date'].tail(num_periods_to_plot), put_obv['obv_bb_upper'].tail(num_periods_to_plot), label='Put OBV Upper Band')
                self.ax[0].plot(put_obv['date'].tail(num_periods_to_plot), put_obv['obv_bb_lower'].tail(num_periods_to_plot), label='Put OBV Lower Band')
                # self.ax[0].plot(put_obv['date'].tail(num_periods_to_plot), put_obv['obv_upper_band2'].tail(num_periods_to_plot), label='Put OBV Upper Band 2')
                # self.ax[0].plot(put_obv['date'].tail(num_periods_to_plot), put_obv['obv_lower_band2'].tail(num_periods_to_plot), label='Put OBV Lower Band 2')
                
                self.ax[0].set_title('Real-Time Options Aggregate On-Balance Volume')
                self.ax[0].set_xlabel('Time')
                self.ax[0].set_ylabel('OBV')
                # self.ax.legend()
                # # hide legen

                # show only the the last n chars in the xaxis tick labels 
                n = 8  # Number of characters to show
                # dont show every single x-axis label
                self.ax[0].set_xticks(self.ax[0].get_xticks()[::5])
                self.ax[0].set_xticklabels([label.get_text()[-n:] for label in self.ax[0].get_xticklabels()])

                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45)

                # ax1_twin = self.ax[1].twinx()
                self.ax[1].clear() 
                self.ax[1].plot(call_obv['date'].tail(num_periods_to_plot), call_obv['obv_call-put-ratio'].tail(num_periods_to_plot), label='Call OBV / Put OBV', color='blue')
                self.ax[1].plot(call_obv['date'].tail(num_periods_to_plot), call_obv['obv_cp_ratio_upper'].tail(num_periods_to_plot), label='Call OBV / Put OBV Upper Band', color='yellow', linestyle='--')
                self.ax[1].plot(call_obv['date'].tail(num_periods_to_plot), call_obv['obv_cp_ratio_lower'].tail(num_periods_to_plot), label='Call OBV / Put OBV Lower Band', color='yellow', linestyle='--')

                # self.ax[2].clear()
                # ax1_twin.clear()

                # self.ax[1].bar(call_obv['date'].tail(num_periods_to_plot), call_obv['obv_pct_change'].tail(num_periods_to_plot), label='Call OBV % Change', color='green')
                # self.ax[2].bar(put_obv['date'].tail(num_periods_to_plot), put_obv['obv_pct_change'].tail(num_periods_to_plot), label='Put OBV % Change', color='red')


                self.fig.tight_layout()

                plt.draw()
            
            ani = FuncAnimation(self.fig, update_plot , interval=60000, cache_frame_data=False)  # Update every minute
            plt.show(block=True) 

# Example usage:
def main(symbol):
    analyzer = OptionsAnalyzer()
    analyzer.connect()
    analyzer.monitor_realtime(symbol, num_periods_to_plot=160)
    # For historical analysis
    # symbol = 'AAPL'
    # options = analyzer.get_otm_options(symbol)
    # all_contracts = options['calls'] + options['puts']
    # hist_data = analyzer.get_historical_data(all_contracts)
    # obv_data = analyzer.calculate_aggregate_obv(hist_data)
    # print("Historical OBV:", obv_data)
    
    # # For real-time monitoring
    # def on_data_update(obv_data):
    #     print("Real-time OBV update:", obv_data.tail(1))
    
    # analyzer.monitor_realtime(symbol, callback=on_data_update)
    # analyzer.ib.run()  # Start event loop

if __name__ == '__main__':
    main()