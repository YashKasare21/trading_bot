import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gemini_utils import get_gemini_sentiment
# Make sure the file is named 'training_colab.py' or update the import
from training_colab import StockTradingEnv, add_signals
import config
import os
import datetime

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
    start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())

    if st.sidebar.button("Run Trading Simulation"):
        with st.spinner('Fetching data, analyzing sentiment, and running simulation...'):
            try:
                # 1. Fetch Data
                data = yf.download(ticker, start=start_date, end=end_date)
                if data.empty:
                    st.error("Could not fetch data for the given ticker and date range.")
                else:
                    # 2. Add Technical Indicators
                    st.write("Adding technical indicators...")
                    # Create a separate DataFrame for the model
                    model_data = add_signals(data.copy())
                    model_data.fillna(0, inplace=True)

                    # 3. Get Sentiment (for display purposes only)
                    news_query = f'{ticker} stock market news today'
                    sentiment_result = get_gemini_sentiment(news_query, api_key=gemini_api_key)
                    st.write(f"**Recent News Sentiment:** {sentiment_result.get('sentiment_label', 'Neutral')}")
                    st.info(f"**Sentiment Analysis Summary:**\n{sentiment_result.get('summary', 'Not available.')}")

                    # 4. Load Model and Run Simulation
                    if os.path.exists(config.MODEL_PATH):
                        model = PPO.load(config.MODEL_PATH)
                        
                        # IMPORTANT FIX: Initialize the environment with the data that
                        # matches the model's training data (without the sentiment score).
                        env = StockTradingEnv(model_data, window_size=config.WINDOW_SIZE, initial_balance=config.INITIAL_BALANCE, transaction_cost_pct=config.TRANSACTION_COST_PCT)
                        obs, info = env.reset()
                        
                        buy_signals, sell_signals = [], []
                        # Loop for the length of the data fed to the environment
                        for i in range(len(model_data) - env.window_size - 1):
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
                        
                        # Plot against the original 'data' DataFrame for correct dates
                        ax.plot(data.index, data['Close'], label='Close Price', alpha=0.7)
                        ax.scatter(data.index[buy_signals], data['Close'].iloc[buy_signals], label='Buy Signal', marker='^', color='green', s=100, zorder=5)
                        ax.scatter(data.index[sell_signals], data['Close'].iloc[sell_signals], label='Sell Signal', marker='v', color='red', s=100, zorder=5)
                        
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
                        st.error(f"Model file not found at {config.MODEL_PATH}. Please train the model first.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter your Gemini API Key to proceed.")