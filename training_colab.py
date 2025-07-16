# training_colab.py

import pandas as pd
import numpy as np
import yfinance as yf
from ta import add_all_ta_features
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ==============================================================================
#  CONFIGURATION VARIABLES
# ==============================================================================
WINDOW_SIZE = 10
INITIAL_BALANCE = 10000
TRANSACTION_COST_PCT = 0.001
MODEL_PATH = "trading_bot_model_PPO.zip"


# ==============================================================================
#  1. DATA PROCESSING FUNCTION
# ==============================================================================
def add_signals(df):
    """
    Adds a comprehensive set of technical analysis features to the dataframe.
    This function now expects a clean DataFrame with only the required OHLCV columns.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicator columns.
    """
    # The function now assumes df is a clean DataFrame with the correct columns.
    # The pd.to_numeric conversion is handled before calling this function.
    df_with_ta = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    return df_with_ta


# ==============================================================================
#  2. CUSTOM TRADING ENVIRONMENT
# ==============================================================================
class StockTradingEnv(gym.Env):
    """
    A stock trading environment for reinforcement learning, compatible with Gymnasium.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=WINDOW_SIZE, initial_balance=INITIAL_BALANCE, transaction_cost_pct=TRANSACTION_COST_PCT):
        super(StockTradingEnv, self).__init__()

        self.df = df.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observations: market data + portfolio state (cash, stock holdings)
        self.num_market_features = self.df.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, self.num_market_features + 2), # +2 for cash and stock value
            dtype=np.float32
        )

    def _get_observation(self):
        frame = self.df.iloc[self.current_step - self.window_size + 1 : self.current_step + 1]
        
        # Market data
        market_obs = frame.values.astype(np.float32)

        # Portfolio data
        cash_normalized = self.cash_balance / self.initial_balance
        current_price = self.df['Close'].iloc[self.current_step]
        stock_value = self.stock_holdings * current_price
        stock_value_normalized = stock_value / self.initial_balance
        
        # Create portfolio state array and broadcast it across the window
        portfolio_state = np.array([[cash_normalized, stock_value_normalized]] * self.window_size, dtype=np.float32)

        # Combine market and portfolio data
        obs = np.hstack((market_obs, portfolio_state))
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size - 1
        self.cash_balance = self.initial_balance
        self.stock_holdings = 0
        self.portfolio_value_history = [self.initial_balance]
        
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        current_price = self.df['Close'].iloc[self.current_step]

        if action == 1:  # Buy
            buy_amount = self.cash_balance * 0.9
            shares_to_buy = buy_amount / current_price
            cost = shares_to_buy * current_price * (1 + self.transaction_cost_pct)
            if self.cash_balance >= cost:
                self.cash_balance -= cost
                self.stock_holdings += shares_to_buy

        elif action == 2:  # Sell
            shares_to_sell = self.stock_holdings * 0.9
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost_pct)
                self.cash_balance += revenue
                self.stock_holdings -= shares_to_sell

        current_portfolio_value = self.cash_balance + self.stock_holdings * current_price
        reward = (current_portfolio_value - self.portfolio_value_history[-1]) / self.portfolio_value_history[-1]
        self.portfolio_value_history.append(current_portfolio_value)

        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, False, info

    def _get_info(self):
        current_price = self.df['Close'].iloc[self.current_step]
        return {
            'portfolio_value': self.cash_balance + self.stock_holdings * current_price,
            'cash': self.cash_balance,
            'stock_holdings': self.stock_holdings
        }

    def render(self, mode='human', close=False):
        pass

# ==============================================================================
#  3. MAIN TRAINING BLOCK
# ==============================================================================
if __name__ == '__main__':
    
    print("Downloading BTC-USD data for training...")
    raw_data = yf.download("BTC-USD", start="2017-01-01", end="2023-12-31", interval="1d")
    
    if raw_data.empty:
        raise ValueError("Failed to download data. The ticker may be invalid or the date range is incorrect.")
    
    # --- ROBUST DATA CLEANING BLOCK ---
    print("Cleaning and preparing data...")
    
    # yfinance can return MultiIndex columns. This is a more robust way to flatten them.
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    # yfinance sometimes returns lowercase columns. Standardize to capitalized format.
    raw_data.columns = [col.capitalize() for col in raw_data.columns]

    # Explicitly select and copy the required columns to avoid SettingWithCopyWarning
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check if all required columns are present after cleaning
    if not all(col in raw_data.columns for col in required_cols):
        raise KeyError(f"One or more required columns not found. Available columns: {raw_data.columns.tolist()}")

    data = raw_data[required_cols].copy()

    # Apply pd.to_numeric to each column individually to ensure they are 1D
    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data.dropna(inplace=True) # Drop rows with NaNs after numeric conversion
    # --- END DATA CLEANING BLOCK ---

    print("Adding technical indicators...")
    # Pass the clean DataFrame to the function
    data = add_signals(data)
    
    print("Adding SIMULATED sentiment scores for training...")
    data['Sentiment_Score'] = np.random.randint(0, 3, size=len(data)) # 0: Negative, 1: Neutral, 2: Positive
    data.dropna(inplace=True)
    print(f"Training data shape: {data.shape}")
    
    # Initialize the environment
    env = StockTradingEnv(data, window_size=WINDOW_SIZE)
    
    # Train the PPO agent
    print("Training PPO agent...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/")
    model.learn(total_timesteps=100000)
    print("Training complete.")
    
    # Save the trained model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")