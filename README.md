# AI Stock Trading Bot

This project implements an AI-powered stock trading bot using Stable-Baselines3, Gymnasium (formerly OpenAI Gym), and Streamlit. The bot is trained in a custom trading environment and can be run as a web application for interactive analysis.

## Features

- **Custom Trading Environment**: A Gymnasium-compatible environment (`StockTradingEnv`) for simulating stock trading.
- **Reinforcement Learning**: Utilizes Stable-Baselines3 for training a PPO agent to learn optimal trading strategies.
- **Data Integration**: Fetches historical stock data using `yfinance`.
- **Technical Indicators**: Incorporates various technical analysis indicators using `ta` library.
- **Sentiment Analysis**: Integrates with the Gemini API for news sentiment analysis.
- **Streamlit Web Application**: Provides an interactive user interface to visualize trading performance and interact with the bot.

## Project Structure

- `training_colab.py`: Contains the custom `StockTradingEnv` and the training script for the PPO agent.
- `streamlit_app.py`: The main Streamlit application for running the trading bot and visualizing results.
- `gemini_utils.py`: Utility functions for interacting with the Gemini API for sentiment analysis.
- `config.py`: Configuration file (e.g., for API keys, though currently handled via Streamlit secrets).
- `requirements.txt`: Lists all Python dependencies.
- `data.csv`: Sample historical stock data (can be replaced by live data fetching).
- `ppo_trading_bot.zip`: Pre-trained model for the trading bot.

## Setup and Installation

1.  **Clone the repository (if not already done):**
    ```bash
    git clone <repository_url>
    cd trading_bot
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/Scripts/activate  # On Windows
    # source venv/bin/activate    # On macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Gemini API Key:**
    This application requires a Gemini API key for sentiment analysis. You can obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey).
    
    When running the Streamlit application, you will be prompted to enter your Gemini API key in the sidebar.

## How to Run

1.  **Ensure your virtual environment is activated.**

2.  **Run the Streamlit application:**
    ```bash
    streamlit run streamlit_app.py
    ```

3.  **Access the application:**
    Your web browser should automatically open to the Streamlit application (usually `http://localhost:8501`). If not, open your browser and navigate to that address.

## Training the Agent (Optional)

If you wish to retrain the PPO agent or modify the training process, you can run the `training_colab.py` script. This script is designed to be run in an environment like Google Colab, but can also be executed locally.

```bash
python training_colab.py
```

**Note**: Training can be computationally intensive and may require a GPU for faster results.

## Troubleshooting

-   **`ImportError: cannot import name 'get_distribution' from 'pkg_resources'`**: This often indicates an issue with `setuptools`. Try upgrading it: `pip install --upgrade setuptools`.
-   **`ImportError: cannot import name 'gym' from 'gymnasium'`**: Ensure `shimmy` is installed (`pip install shimmy`) and your environment is correctly configured for Gymnasium.
-   **`yfinance` data fetching issues**: Verify your internet connection. Try different ticker symbols and date ranges. `yfinance` can sometimes have intermittent issues.
-   **`'str' object has no attribute 'get'`**: This error typically occurs if the sentiment analysis function returns a string when a dictionary is expected. Ensure `gemini_utils.py`'s `get_gemini_sentiment` returns a dictionary with `sentiment_label`, `sentiment_score`, and `summary`.
-   **Observation Shape Mismatch**: If you encounter errors related to unexpected observation shapes, check the `num_total_features` calculation in `StockTradingEnv`'s `__init__` method and ensure it matches the actual number of features in your `data` DataFrame after technical indicators and sentiment scores are added.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.