# Stock Price Prediction Using LSTM

## Overview
This project uses Long Short-Term Memory (LSTM) neural networks to predict the stock prices of a given company. The model uses historical stock data to train the network and make future predictions. The goal is to assist in forecasting future stock prices based on historical trends and technical indicators like Moving Averages and Relative Strength Index (RSI).

## Features
- **Stock Price Prediction**: Predict future stock prices using LSTM.
- **Technical Indicators**: The model integrates various technical indicators, including:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
- **Data Visualization**: Plots the predicted stock prices against the actual historical stock prices.
- **Model Evaluation**: Evaluate the model on test data and report the loss.

## Requirements
This project requires the following Python libraries:
- `numpy`
- `pandas`
- `yfinance` (for downloading stock data)
- `tensorflow` (for building the LSTM model)
- `scikit-learn` (for data scaling)
- `matplotlib` (for plotting graphs)

To install the required libraries, you can use the following command:

```bash
pip install -r requirements.txt

# file Structure :- 
Stock-Predict/
│
├── assets/                # Folder for storing assets (e.g., graphs, images)
├── data.csv               # The historical stock data in CSV format
├── main-agent.py          # Main script for training the model
├── main-predict.py        # Script for making predictions on stock data
├── README.md              # Project description and instructions
└── stock_model.h5         # Saved LSTM model after training
How to Use
1. Clone the repository
You can clone this repository using the following command:


git clone https://github.com/Singhji007/Stock-Predict.git
cd Stock-Predict

Run This Command :-
python main-predict.py

Future Improvements
Model Tuning: Experiment with different hyperparameters to improve model accuracy.
Additional Features: Integrate more technical indicators (e.g., Bollinger Bands, MACD) for better prediction.
Real-time Data: Incorporate live data and real-time predictions for active trading.


Acknowledgments
TensorFlow for building the deep learning model.
Yahoo Finance (yfinance) for easy access to stock data.
Matplotlib for data visualization.


