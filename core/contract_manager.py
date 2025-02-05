from typing import Dict, List, Tuple
from ib_insync import Stock, Option, Contract, IB
import logging

class OptionsContractManager:
    """
    Manages retrieval and filtering of options contracts
    """
    
    def __init__(self, ib_connection: IB):
        """
        Initialize with an Interactive Brokers connection
        
        Args:
            ib_connection (IB): Active IB connection
        """
        self.ib = ib_connection
        self.logger = logging.getLogger(__name__)
    
    def get_nth_expiry(self, chains, expiry_distance: int = 0) -> str:
        """
        Get the nth nearest expiration date from option chains
        
        Args:
            chains: Option chain parameters
            expiry_distance (int): How far out to look for expiry
        
        Returns:
            str: Expiration date string
        """
        expirations = sorted(set(contract_exp for contract_exp in chains.expirations))
        return expirations[expiry_distance] if expirations else None
    
    def get_strike_bounds(self, 
                          stock: Contract) -> Tuple[float, List[float]]:
        """
        Determine stock price and available strike prices
        
        Args:
            stock (Contract): Stock contract
        
        Returns:
            Tuple of last price and sorted strikes
        """
        try:
            [ticker] = self.ib.reqTickers(stock)
            last_price = ticker.last if ticker.last > 0 else ticker.close
            
            self.logger.info(f"Underlying price: {last_price}")
            
            # Get option chains
            chains = self.ib.reqSecDefOptParams(
                stock.symbol, '', stock.secType, stock.conId
            )
            
            if not chains:
                raise ValueError(f'No option chains found for {stock.symbol}')
            
            chain = next(iter(chains))
            strikes = sorted(strike for strike in chain.strikes if strike % 1 == 0)
            
            return last_price, strikes
        
        except Exception as e:
            self.logger.error(f"Error getting strike bounds: {e}")
            raise
    
    def get_otm_options(self, 
                        symbol: str, 
                        num_strikes: int = 5, 
                        expiry_distance: int = 0) -> Dict[str, List[Contract]]:
        """
        Retrieve Out-of-the-Money (OTM) options
        
        Args:
            symbol (str): Stock symbol
            num_strikes (int): Number of strikes from ATM
            expiry_distance (int): Expiry date offset
        
        Returns:
            Dict of call and put contracts
        """
        try:
            stock = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(stock)
            
            last_price, strikes = self.get_strike_bounds(stock)
            
            # Find ATM strike index
            atm_idx = min(range(len(strikes)), 
                          key=lambda i: abs(strikes[i] - last_price))
            
            chains = self.ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
            expiry = self.get_nth_expiry(chains[0], expiry_distance)
            
            # Create OTM option contracts
            otm_calls = [
                Option(symbol, expiry, strike, 'C', 'SMART', includeExpired=True)
                for strike in strikes[atm_idx:atm_idx+num_strikes]
            ]
            
            otm_puts = [
                Option(symbol, expiry, strike, 'P', 'SMART', includeExpired=True)
                for strike in strikes[atm_idx-num_strikes:atm_idx]
            ]
            
            return {'calls': otm_calls, 'puts': otm_puts}
        
        except Exception as e:
            self.logger.error(f"Error retrieving OTM options: {e}")
            raise