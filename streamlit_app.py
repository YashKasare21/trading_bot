import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gemini_utils import get_gemini_sentiment
# Make sure the file is named 'training_colab.py' or update the import
from training_colab import StockTradingEnv, add_signals
import os
import datetime

# ==============================================================================
#  CONFIGURATION VARIABLES (This solves the ModuleNotFoundError)
# ==============================================================================
WINDOW_SIZE = 10
INITIAL_BALANCE = 10000
TRANSACTION_COST_PCT = 0.001
MODEL_PATH = "trading_bot_model_PPO.zip"


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="AI Trading Bot")
st.title("AI-Powered Trading Bot with Sentiment Analysis")

# --- Sidebar for Inputs ---
st.sidebar.header("Configuration")
gemini_api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

# --- Main App Logic ---
if gemini_api_key:
    st.sidebar.success("Gemini API Key Loaded!")
    ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
    start_date = st.sidebar.date_input("Start Date", datetime.date(2024, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())

    if st.sidebar.button("Run Trading Simulation"):
        with st.spinner('Fetching data, analyzing sentiment, and running simulation...'):
            try:
                # 1. Fetch Data
                raw_data = yf.download(ticker, start=start_date, end=end_date)
                if raw_data.empty:
                    st.error("Could not fetch data for the given ticker and date range.")
                else:
                    # --- ROBUST DATA CLEANING (Mirrors training script) ---
                    st.write("Cleaning and preparing data...")
                    if isinstance(raw_data.columns, pd.MultiIndex):
                        raw_data.columns = raw_data.columns.get_level_values(0)
                    raw_data.columns = [col.capitalize() for col in raw_data.columns]
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if not all(col in raw_data.columns for col in required_cols):
                        raise KeyError(f"One or more required columns not found after cleaning. Available: {raw_data.columns.tolist()}")
                    
                    data_for_plotting = raw_data.copy() # Keep original data for the plot
                    model_data = raw_data[required_cols].copy()
                    for col in required_cols:
                        model_data[col] = pd.to_numeric(model_data[col], errors='coerce')
                    model_data.dropna(inplace=True)
                    # --- END CLEANING ---

                    # 2. Add Technical Indicators
                    st.write("Adding technical indicators...")
                    model_data = add_signals(model_data)

                    # 3. Get Sentiment
                    news_query = f'{ticker} stock market news today'
                    sentiment_result = get_gemini_sentiment(news_query, api_key=gemini_api_key)
                    sentiment_score = sentiment_result.get('sentiment_score', 1) # Default to neutral (1)
                    st.write(f"**Recent News Sentiment:** {sentiment_result.get('sentiment_label', 'Neutral')}")
                    st.info(f"**Sentiment Analysis Summary:**\n{sentiment_result.get('summary', 'Not available.')}")

                    # =========================================================
                    #  CRITICAL FIX: Add the sentiment score to the model's data
                    # =========================================================
                    model_data['Sentiment_Score'] = sentiment_score
                    model_data.fillna(0, inplace=True) # Fill any remaining NaNs
                    
                    # 4. Load Model and Run Simulation
                    if os.path.exists(MODEL_PATH):
                        model = PPO.load(MODEL_PATH)
                        
                        env = StockTradingEnv(model_data, window_size=WINDOW_SIZE, initial_balance=INITIAL_BALANCE, transaction_cost_pct=TRANSACTION_COST_PCT)
                        obs, info = env.reset()
                        
                        buy_signals, sell_signals = [], []
                        # Loop for the length of the data fed to the environment
                        for i in range(len(model_data) - env.window_size -1):
                            action, _ = model.predict(obs, deterministic=True)
                            obs, reward, done, _, info = env.step(action)
                            
                            # Adjust signal index to match original dataframe
                            signal_index = env.window_size + i + 1
                            
                            if action == 1: # Buy
                                buy_signals.append(signal_index)
                            elif action == 2: # Sell
                                sell_signals.append(signal_index)
                            if done:
                                break
                        
                        # 5. Display Results
                        st.subheader(f"Trading Simulation for {ticker}")
                        fig, ax = plt.subplots(figsize=(15, 7))
                        
                        # Plot against the original 'data_for_plotting' DataFrame for correct dates
                        ax.plot(data_for_plotting.index, data_for_plotting['Close'], label='Close Price', alpha=0.7)
                        
                        # Ensure signal indices are within bounds before plotting
                        valid_buy_signals = [s for s in buy_signals if s < len(data_for_plotting)]
                        valid_sell_signals = [s for s in sell_signals if s < len(data_for_plotting)]

                        if valid_buy_signals:
                            ax.scatter(data_for_plotting.index[valid_buy_signals], data_for_plotting['Close'].iloc[valid_buy_signals], label='Buy Signal', marker='^', color='green', s=100, zorder=5)
                        if valid_sell_signals:
                            ax.scatter(data_for_plotting.index[valid_sell_signals], data_for_plotting['Close'].iloc[valid_sell_signals], label='Sell Signal', marker='v', color='red', s=100, zorder=5)
                        
                        ax.set_title(f'{ticker} Trading Signals')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Price (USD)')
                        ax.legend()
                        st.pyplot(fig)

                        st.subheader("Simulation Performance")
                        final_value = info.get('portfolio_value', 0)
                        initial_balance = env.initial_balance
                        returns = ((final_value - initial_balance) / initial_balance) * 100
                        st.metric("Final Portfolio Value", f"${final_value:,.2f}")
                        st.metric("Total Return", f"{returns:.2f}%")

                    else:
                        st.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter your Gemini API Key to proceed.")
