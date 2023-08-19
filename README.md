# Stock_Predictor_V2

Stock_Predictor_V2 is a Python-based project that leverages data extraction, data engineering, regression modeling, time series analysis, and reinforcement learning to predict and analyze stock market data. This project aims to provide insights into stock price trends, make predictions, and even includes a basic trading bot.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Overview

The Stock_Predictor_V2 project is divided into several key components:

1. **Data Extraction and Data Engineering:**
   - Utilizes the `yfinance` module to efficiently collect historical market data, such as stock prices and financial information, from Yahoo Finance.
   - Performs data engineering tasks, including feature extraction and trend analysis.

2. **Regression Modeling:**
   - Applies linear and polynomial regression models to predict stock price changes.
   - Determines the optimal model complexity based on evaluation metrics like mean absolute percentage error (MAPE).

3. **Time Series Analysis:**
   - Explores LSTM and CNN neural network models for time series analysis of stock data.
   - Trains models to capture patterns and make predictions.

4. **Reinforcement Learning and Trading Bot:**
   - Develops a basic trading bot using reinforcement learning techniques.
   - Utilizes OpenAI Gym and Stable Baselines libraries for RL agent training.

## Project Structure

The project's directory structure is organized as follows:

- `models/`: Stores trained regression and reinforcement learning models.
- `Predictions_code.py`: Python script containing the core functionality of the project.
- `README.md`: This readme file.
- `requirements.txt`: Lists the project's dependencies.

## Getting Started

To get started with the Stock_Predictor_V2 project, follow these steps:

1. Clone this repository to your local machine:
   ```shell
   git clone https://github.com/your-username/Stock_Predictor_V2.git
   ```

2. Install the project's dependencies using pip:
   ```shell
   pip install -r requirements.txt
   ```

3. Ensure you have Python installed, preferably Python 3.7 or higher.

## Usage

To use the Stock_Predictor_V2 project, follow these usage instructions:

1. Run the main Python script, e.g., `main_gui.py`, to interact with the project's functionalities.

2. Select a stock model or enter "Other" to create a new model by providing a stock ticker symbol.

3. Explore the generated results, including regression analysis, time series predictions, and trading bot performance.

## Results

The project provides insights into stock market data analysis and prediction. However, the performance of the trading bot may vary, and further enhancements can be made for more accurate trading strategies.

## Contributing

Contributions to Stock_Predictor_V2 are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Commit your changes and create a pull request.
