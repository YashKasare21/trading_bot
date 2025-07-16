# AI Stock Trading Bot

# AI-Powered Stock Trading Bot

This project develops an advanced AI-powered stock trading bot leveraging **Stable-Baselines3** for reinforcement learning, **Gymnasium** (formerly OpenAI Gym) for environment simulation, and **Streamlit** for an interactive web application. The bot is designed to learn optimal trading strategies within a custom-built trading environment and provides a user-friendly interface for real-time analysis and simulation.

## Key Features

-   **Custom Trading Environment**: A robust, Gymnasium-compatible `StockTradingEnv` for realistic stock market simulations.
-   **Reinforcement Learning Integration**: Employs Stable-Baselines3 to train a Proximal Policy Optimization (PPO) agent, enabling the bot to develop sophisticated trading strategies.
-   **Dynamic Data Acquisition**: Seamlessly fetches real-time and historical stock data via the `yfinance` library.
-   **Comprehensive Technical Analysis**: Integrates the `ta` library to generate a wide array of technical indicators, enriching the observation space for the AI.
-   **Advanced Sentiment Analysis**: Incorporates the Gemini API to analyze news sentiment, providing crucial market insights.
-   **Interactive Web Application**: A user-friendly Streamlit interface for visualizing trading performance, conducting backtests, and interacting with the AI bot.

## Project Structure

-   `training_colab.py`: Defines the custom `StockTradingEnv` and includes the script for training the PPO agent.
-   `streamlit_app.py`: The core Streamlit application, providing the interactive user interface for running simulations and visualizing trading outcomes.
-   `gemini_utils.py`: Contains essential utility functions for integrating and interacting with the Gemini API for sentiment analysis.
-   `config.py`: Centralized configuration file managing key parameters such as API keys, default dates, and model paths.
-   `requirements.txt`: Lists all necessary Python dependencies for the project.
-   `data.csv`: A sample dataset for initial setup and testing (the application primarily uses live data fetching).
-   `trading_bot_model_PPO.zip`: The pre-trained PPO model, ready for deployment within the Streamlit application.

## Setup and Installation

Follow these steps to set up and run the project locally:

1.  **Clone the Repository**
    If you haven't already, clone the project repository to your local machine:
    ```bash
    git clone https://github.com/YashKasare21/trading_bot.git
    cd trading_bot
    ```

2.  **Create and Activate a Virtual Environment**
    It is highly recommended to use a virtual environment to manage dependencies:
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Install all required Python packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Gemini API Key**
    This application utilizes the Gemini API for sentiment analysis. Obtain your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    
    During the Streamlit application runtime, you will be prompted to enter your Gemini API key directly within the sidebar. This ensures secure handling of your credentials.

## How to Run the Application

Once the setup is complete, you can launch the Streamlit web application:

1.  **Activate your virtual environment** (if not already active).

2.  **Execute the Streamlit command** from the project root directory:
    ```bash
    streamlit run streamlit_app.py
    ```

3.  **Access the Application**: Your default web browser should automatically open to the Streamlit interface (typically at `http://localhost:8501`). If it doesn't, manually navigate to this URL.

## Training the AI Agent (Optional)

For users interested in retraining the PPO agent or customizing the training methodology, the `training_colab.py` script is provided. While optimized for environments like Google Colab, it can also be executed locally:

```bash
python training_colab.py
```

**Note**: Agent training is a computationally intensive process. Utilizing a GPU is highly recommended to significantly accelerate training times.

## Troubleshooting Common Issues

-   **`ImportError: cannot import name 'get_distribution' from 'pkg_resources'`**:
    This error often points to an outdated `setuptools` package. Resolve it by upgrading:
    ```bash
    pip install --upgrade setuptools
    ```

-   **`ImportError: cannot import name 'gym' from 'gymnasium'`**:
    Ensure that `shimmy` is installed and your Gymnasium environment is correctly configured:
    ```bash
    pip install shimmy
    ```

-   **`yfinance` Data Fetching Problems**:
    Verify your internet connection. Issues can sometimes arise from `yfinance` itself; try alternative ticker symbols or adjust date ranges.

-   **`'str' object has no attribute 'get'`**:
    This error suggests that the `get_gemini_sentiment` function in `gemini_utils.py` is returning a string instead of the expected dictionary. Confirm that the function's output includes `sentiment_label`, `sentiment_score`, and `summary` keys.

-   **Observation Shape Mismatch Errors**:
    If you encounter errors related to inconsistent observation shapes, review the `num_total_features` calculation within the `StockTradingEnv`'s `__init__` method. Ensure it accurately reflects the total number of features in your `data` DataFrame, especially after the integration of technical indicators and sentiment scores.

## License

This project is distributed under the MIT License. See the `LICENSE` file for more details.