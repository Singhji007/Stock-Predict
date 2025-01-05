import sys
import warnings
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from tqdm import tqdm
from pandas_datareader import data as pdr
import yfinance as yf
import os

# Suppress warnings for cleaner output
if not sys.warnoptions:
    warnings.simplefilter("ignore")

sns.set()
tf.random.set_seed(1234)
yf.pdr_override()

# Argument parser setup
parser = argparse.ArgumentParser(description="Train Stock Market Predictor")
parser.add_argument("--symbol", type=str, required=True, help="Stock symbol to use")
parser.add_argument("--period", type=str, default="2y", help="Data period to download")
parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
parser.add_argument("--sims", type=int, default=5, help="Number of simulations")
args = parser.parse_args()

# Download stock data
df = pdr.get_data_yahoo(args.symbol, period=args.period)
df.to_csv("data.csv")
df = pd.read_csv("data.csv")

# Preprocess data
scaler = MinMaxScaler().fit(df[["Close"]].astype("float32"))
df_log = scaler.transform(df[["Close"]].astype("float32"))
df_log = pd.DataFrame(df_log)

# Hyperparameters
SIMULATION_SIZE = args.sims
NUM_LAYERS = 1
SIZE_LAYER = 128
TIMESTAMP = 5
EPOCHS = args.epochs
DROPOUT_RATE = 0.8
TEST_SIZE = 30
LEARNING_RATE = 0.01

df_train = df_log
print("Data shape:", df.shape, "Training data shape:", df_train.shape)

# Define the LSTM model
class StockPredictorModel(tf.keras.Model):
    def __init__(self, num_layers, size_layer, output_size, dropout_rate):
        super(StockPredictorModel, self).__init__()
        self.lstm_layers = [
            tf.keras.layers.LSTM(size_layer, return_sequences=True, return_state=True)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(output_size)

    def call(self, inputs, training=False, initial_states=None):
        x = inputs
        states = initial_states
        for lstm_layer in self.lstm_layers:
            x, *states = lstm_layer(x, initial_state=states)
        if training:
            x = self.dropout(x)
        x = self.dense(x)
        return x, states

# Helper functions
def calculate_accuracy(real, predicted):
    real, predicted = np.array(real) + 1, np.array(predicted) + 1
    return (1 - np.sqrt(np.mean(np.square((real - predicted) / real)))) * 100

def anchor(signal, weight):
    smoothed = []
    last = signal[0]
    for val in signal:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# Training and forecasting
def forecast():
    model = StockPredictorModel(NUM_LAYERS, SIZE_LAYER, df_log.shape[1], DROPOUT_RATE)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(inputs, targets, states):
        with tf.GradientTape() as tape:
            predictions, states = model(inputs, training=True, initial_states=states)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, states

    # Prepare training data
    date_ori = pd.to_datetime(df["Date"]).tolist()
    init_states = [tf.zeros((1, SIZE_LAYER)) for _ in range(NUM_LAYERS * 2)]
    
    for epoch in tqdm(range(EPOCHS), desc="Training"):
        total_loss = []
        for i in range(0, len(df_train) - TIMESTAMP, TIMESTAMP):
            batch_x = tf.expand_dims(df_train.iloc[i:i+TIMESTAMP].values, axis=0)
            batch_y = tf.expand_dims(df_train.iloc[i+1:i+TIMESTAMP+1].values, axis=0)
            loss, init_states = train_step(batch_x, batch_y, init_states)
            total_loss.append(loss.numpy())
        tqdm.write(f"Epoch {epoch+1}: Loss = {np.mean(total_loss)}")
    
    # Forecast future values
    future_predictions = []
    current_states = init_states
    input_data = tf.expand_dims(df_train[-TIMESTAMP:].values, axis=0)

    for _ in range(TEST_SIZE):
        prediction, current_states = model(input_data, training=False, initial_states=current_states)
        future_predictions.append(prediction.numpy()[0, -1, 0])
        input_data = tf.expand_dims(np.append(input_data.numpy()[0, 1:], [[future_predictions[-1]]], axis=0), axis=0)
        date_ori.append(date_ori[-1] + timedelta(days=1))
    
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)), date_ori

# Run simulations
results = []
for sim in range(SIMULATION_SIZE):
    print(f"Simulation {sim+1}")
    prediction, dates = forecast()
    results.append(prediction.flatten())

# Filter valid results
valid_results = [
    result for result in results
    if (result[-TEST_SIZE:] < df["Close"].min()).sum() == 0
    and (result[-TEST_SIZE:] > df["Close"].max() * 2).sum() == 0
]

accuracies = [
    calculate_accuracy(df["Close"].values, result[:-TEST_SIZE])
    for result in valid_results
]

# Plot results
plt.figure(figsize=(15, 5))
for i, result in enumerate(valid_results):
    plt.plot(result, label=f"Forecast {i+1}")
plt.plot(df["Close"], label="True Trend", color="black")
plt.legend()
plt.title(f"Stock: {args.symbol} | Avg. Accuracy: {np.mean(accuracies):.2f}%")
plt.xticks(np.arange(0, len(dates), 30), dates[::30], rotation=45)
plt.tight_layout()
plt.show()

# Clean up
os.remove("data.csv")
