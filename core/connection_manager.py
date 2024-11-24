from typing import Optional
from ib_insync import IB
import logging

class IBConnectionManager:
    """
    Manages connection to Interactive Brokers API
    
    Attributes:
        connection (IB): Interactive Brokers connection instance
        logger (logging.Logger): Logging instance for tracking connection events
    """
    
    def __init__(self, 
                 host: str = '127.0.0.1', 
                 port: int = 7496, 
                 client_id: int = 10):
        """
        Initialize IB connection parameters
        
        Args:
            host (str): IB Gateway/TWS host address
            port (int): Connection port
            client_id (int): Unique client identifier
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connection: Optional[IB] = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> IB:
        """
        Establish connection to Interactive Brokers
        
        Returns:
            IB: Connected IB instance
        
        Raises:
            ConnectionError: If unable to establish connection
        """
        try:
            self.connection = IB()
            self.connection.connect(self.host, self.port, self.client_id)
            self.logger.info(f"Connected to IB at {self.host}:{self.port}")
            return self.connection
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            raise ConnectionError(f"Could not connect to IB: {e}")
    
    def disconnect(self):
        """
        Safely disconnect from Interactive Brokers
        """
        if self.connection:
            try:
                self.connection.disconnect()
                self.logger.info("Disconnected from Interactive Brokers")
            except Exception as e:
                self.logger.warning(f"Error during disconnection: {e}")
            finally:
                self.connection = None
    
    def __enter__(self):
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()