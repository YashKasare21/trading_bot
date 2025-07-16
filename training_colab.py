# Step 1 :
# !pip install gym stable-baselines3 yfinance pandas ta streamlit gym-anytrading finrl requests shimmy gymnasium

# Step 2: Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import config # Added this line
from gemini_utils import get_gemini_sentiment # Assuming gemini_utils.py is in the same directory


# Step 3: Download BTC-USD data
print("Downloading BTC-USD data...")
data = yf.download("BTC-USD", start="2024-01-01", interval="1h", auto_adjust=False)
print("Data downloaded successfully.")

# Drop the 'Ticker' level from the MultiIndex and select required columns
if isinstance(data.columns, pd.MultiIndex) and 'Ticker' in data.columns.names:
    data.columns = data.columns.droplevel('Ticker')
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Ensure the required columns exist after droplevel
    data = data[required_cols]
elif not isinstance(data.columns, pd.MultiIndex):
     # If not a MultiIndex, just select required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[required_cols]
else:
    print("Warning: Unexpected column structure.")




# Step 4: Add technical indicators
print("Adding technical indicators...")

# Ensure required columns are numeric and fill any resulting NaNs
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0) # Fill NaNs with 0

# *** Add data inspection steps ***
print("\nData Types of Required Columns Before TA Calculation:")
print(data[required_cols].dtypes)

print("\nChecking for Non-Numeric/NaNs After Cleaning in Required Columns:")
print(data[required_cols].apply(lambda x: pd.to_numeric(x, errors='coerce').isna().any()))
# *** End data inspection steps ***

# Add technical indicators using the flattened and selected column names
data = add_all_ta_features(
    data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
)

# Add sentiment score (simulated for training)
data['Sentiment_Score'] = np.random.randint(0, 3, size=len(data)) # 0: Negative, 1: Neutral, 2: Positive
data.dropna(inplace=True)
print("Technical indicators and sentiment scores added.")


# Step 5: Create custom Gym environment
class StockTradingEnv(gym.Env):
    # Add render_mode to metadata for Gymnasium compatibility
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=config.WINDOW_SIZE, initial_balance=config.INITIAL_BALANCE, transaction_cost_pct=config.TRANSACTION_COST_PCT):
        super(StockTradingEnv, self).__init__()

        self.df = df.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct

        self.action_space = spaces.Discrete(3) # Hold, Buy, Sell

        self.num_market_features = self.df.shape[1]
        self.num_total_features = self.num_market_features + 2 # Add cash and stock holdings
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.num_total_features), dtype=np.float32)

        self.reset()

    def _get_observation(self):
        frame_start = self.current_step - self.window_size + 1
        frame_end = self.current_step + 1

        if frame_start < 0:
            padded_df = pd.DataFrame(0.0, index=range(abs(frame_start)), columns=self.df.columns)
            current_frame_data = pd.concat([padded_df, self.df.iloc[:frame_end]])
        else:
            current_frame_data = self.df.iloc[frame_start:frame_end]

        obs_array = current_frame_data.values.astype(np.float32)

        normalized_cash = np.array([self.cash_balance / self.initial_balance]).astype(np.float32)
        current_price = self.df['Close'].iloc[self.current_step] # Get current price here
        current_stock_value = self.stock_holdings * current_price
        normalized_stock_value = np.array([current_stock_value / self.initial_balance]).astype(np.float32)

        portfolio_state = np.zeros((self.window_size, 2))
        # Ensure normalized_cash and normalized_stock_value are 1-dimensional for concatenation
        normalized_cash_array = np.array([normalized_cash]).reshape(1,)
        normalized_stock_value_array = np.array([normalized_stock_value]).reshape(1,)

        # Concatenate the two values into a single array for assignment
        # This creates a 1D array of shape (2,)
        portfolio_state_values = np.concatenate((normalized_cash_array, normalized_stock_value_array))

        # Assign the 1D array to the last row of portfolio_state
        # Ensure portfolio_state[-1] is a 1D slice that can accept a 1D array
        # If portfolio_state is (window_size, num_features), then portfolio_state[-1] is (num_features,)
        # We need to ensure portfolio_state_values has the same shape as portfolio_state[-1]
        # Assuming portfolio_state[-1] is meant to hold these two values, its shape should be (2,)
        # If portfolio_state is (window_size, 2), then this assignment is correct.
        # If portfolio_state is (window_size, N) where N > 2, then we need to decide where to put these.
        # Based on the original code, it seems portfolio_state[-1] is intended to be replaced by these two values.
        # Let's assume portfolio_state is (window_size, 2) or we are only replacing the first two elements of the last row.
        # For now, let's assume portfolio_state[-1] is of shape (2,)
        portfolio_state[-1] = portfolio_state_values

        # Ensure obs_with_portfolio has the correct shape after hstack
        # obs_array is (window_size, num_features - 2)
        # portfolio_state is (window_size, 2)

        # hstack will result in (window_size, num_features)
        obs_with_portfolio = np.hstack([obs_array, portfolio_state])

        # Ensure the observation has the correct shape (window_size, num_total_features)
        # This might be needed if padding changes the shape unexpectedly
        expected_shape = (self.window_size, self.num_total_features)
        if obs_with_portfolio.shape != expected_shape:
             # If shape is incorrect, attempt to resize or pad
             # A simple approach is to pad with zeros if the first dimension is smaller
             if obs_with_portfolio.shape[0] < self.window_size:
                 padding_needed = self.window_size - obs_with_portfolio.shape[0]
                 padding_shape = (padding_needed, self.num_total_features)
                 padding_array = np.zeros(padding_shape, dtype=np.float32)
                 obs_with_portfolio = np.vstack([padding_array, obs_with_portfolio])
             # If the first dimension is larger, this might indicate an issue with frame_start/frame_end logic
             # For now, we assume padding is the primary issue at the start of the episode
             elif obs_with_portfolio.shape[0] > self.window_size:
                 # This case indicates a potential error in frame slicing or padding logic
                 # For robustness, you might want to log a warning or raise an error
                 print(f"Warning: Observation shape {obs_with_portfolio.shape} is larger than expected {expected_shape}.")
                 # You might need to slice to the correct shape, depending on desired behavior
                 obs_with_portfolio = obs_with_portfolio[-self.window_size:, :]


        return obs_with_portfolio


    def reset(self, seed=None, options=None):
        # For Gymnasium compatibility, reset should return (observation, info)
        super().reset(seed=seed)
        self.current_step = self.window_size - 1
        self.cash_balance = self.initial_balance
        self.stock_holdings = 0
        self.portfolio_value_history = [self.initial_balance]
        self.total_shares_bought = 0
        self.total_shares_sold = 0

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # For Gymnasium compatibility, step should return (observation, reward, terminated, truncated, info)
        self.current_step += 1
        done = False
        reward = 0

        if self.current_step >= len(self.df) -1:
            done = True
            final_portfolio_value = self.cash_balance + self.stock_holdings * self.df['Close'].iloc[-1]
            reward = (final_portfolio_value - self.initial_balance) / self.initial_balance
            self.portfolio_value_history.append(final_portfolio_value)

            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, done, False, info

        current_price = self.df['Close'].iloc[self.current_step]

        if action == 1:  # Buy
            buy_amount = self.cash_balance * 0.9
            shares_to_buy = buy_amount / current_price
            transaction_cost = shares_to_buy * current_price * self.transaction_cost_pct

            if self.cash_balance >= (buy_amount + transaction_cost):
                self.cash_balance -= (buy_amount + transaction_cost)
                self.stock_holdings += shares_to_buy
                self.total_shares_bought += shares_to_buy
        elif action == 2:  # Sell
            sell_shares = self.stock_holdings * 0.9
            if sell_shares > 0:
                transaction_cost = sell_shares * current_price * self.transaction_cost_pct
                self.cash_balance += (sell_shares * current_price) - transaction_cost
                self.stock_holdings -= sell_shares
                self.total_shares_sold += sell_shares

        current_portfolio_value = self.cash_balance + self.stock_holdings * current_price
        self.portfolio_value_history.append(current_portfolio_value)
        reward = (current_portfolio_value - self.portfolio_value_history[-2]) / self.portfolio_value_history[-2]

        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, done, False, info

    def _get_info(self):
        # Ensure current_step is within bounds for iloc
        current_price = self.df['Close'].iloc[min(self.current_step, len(self.df) - 1)]
        return {
            'portfolio_value': self.cash_balance + self.stock_holdings * current_price,
            'cash': self.cash_balance,
            'stock_holdings': self.stock_holdings
        }


    def render(self, mode='human', close=False):
        pass

print("Custom Gym environment created.")



# Step 6: Train PPO agent
# Ensure data has enough rows for the window size and 'Close' column exists
WINDOW_SIZE = 10 # This should match the default or specified window_size in StockTradingEnv
if 'Close' not in data.columns:
    raise ValueError("DataFrame must contain a 'Close' column for StockTradingEnv.")
if len(data) < WINDOW_SIZE:
    raise ValueError(f"DataFrame has {len(data)} rows, but StockTradingEnv requires at least {WINDOW_SIZE} rows.")

env = StockTradingEnv(data, window_size=WINDOW_SIZE)

print("Training PPO agent...")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
print("Training complete.")


# Step 7: Save model and data
print("Saving model and data...")
model.save("ppo_trading_bot")
data.to_csv("data.csv")
print("Files saved as ppo_trading_bot.zip and data.csv")