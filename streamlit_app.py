import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gemini_utils import get_gemini_sentiment
from training_colab import StockTradingEnv # Import the environment from the training script
import config
import os
import datetime # Import datetime module

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
    start_date = st.sidebar.date_input("Start Date", datetime.datetime.strptime(config.DEFAULT_START_DATE, "%Y-%m-%d").date())
    end_date = st.sidebar.date_input("End Date", datetime.datetime.strptime(config.DEFAULT_END_DATE, "%Y-%m-%d").date())

    if st.sidebar.button("Run Trading Simulation"):
        with st.spinner('Fetching data, analyzing sentiment, and running simulation...'):
            try:
                # 1. Fetch Data
                st.write(f"Debug: Ticker={ticker}, Start Date={start_date}, End Date={end_date}")
                data = yf.download(ticker, start=start_date, end=end_date)
                if data.empty:
                    st.error("Could not fetch data for the given ticker and date range.")
                else:
                    # 2. Get Sentiment
                    news_query = f'{ticker} stock market news'
                    sentiment_result = get_gemini_sentiment(news_query, api_key=gemini_api_key)
                    sentiment_score = sentiment_result.get('sentiment_score', 1) # Default to neutral
                    st.write(f"**Recent News Sentiment:** {sentiment_result.get('sentiment_label', 'Neutral')}")
                    st.info(f"**Sentiment Analysis Summary:**\n{sentiment_result.get('summary', 'Not available.')}")

                    # 3. Prepare data for the environment
                    data['Sentiment_Score'] = sentiment_score
                    data.ffill(inplace=True) # Forward fill to handle any missing values

                    # 4. Load Model and Run Simulation
                    if os.path.exists(config.MODEL_PATH):
                        model = PPO.load(config.MODEL_PATH)
                        env = StockTradingEnv(data, window_size=config.WINDOW_SIZE, initial_balance=config.INITIAL_BALANCE, transaction_cost_pct=config.TRANSACTION_COST_PCT)
                        obs, info = env.reset()
                        
                        buy_signals, sell_signals = [], []
                        for _ in range(len(data) - 1):
                            action, _ = model.predict(obs, deterministic=True)
                            obs, reward, done, _, info = env.step(action)
                            if action == 1: # Buy
                                buy_signals.append(env.current_step)
                            elif action == 2: # Sell
                                sell_signals.append(env.current_step)
                            if done:
                                break
                        
                        # 5. Display Results
                        st.subheader(f"Trading Simulation for {ticker}")
                        fig, ax = plt.subplots(figsize=(15, 7))
                        ax.plot(data.index, data['Close'], label='Close Price', alpha=0.7)
                        ax.scatter(data.index[buy_signals], data['Close'].iloc[buy_signals], label='Buy Signal', marker='^', color='green', s=100)
                        ax.scatter(data.index[sell_signals], data['Close'].iloc[sell_signals], label='Sell Signal', marker='v', color='red', s=100)
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
                        st.error(f"Model file not found at {MODEL_FILE}. Please train the model first.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter your Gemini API Key to proceed.")