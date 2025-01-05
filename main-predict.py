import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Function to calculate RSI (Relative Strength Index)
def compute_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to forecast future prices
def forecast_future(data, model, scaler, steps, feature_count):
    last_sequence = data[-TIMESTEP:]  # Use the last TIMESTEP of data for forecasting
    future_predictions = []

    for _ in range(steps):
        # Predict the next value
        pred = model.predict(last_sequence.reshape(1, TIMESTEP, feature_count))
        future_predictions.append(pred[0, 0])  # Save the predicted 'Close' value

        # Pad the prediction with zeros to match feature dimensions
        pred_padded = np.concatenate([pred, np.zeros((1, feature_count - 1))], axis=1)

        # Append the prediction and shift the sequence
        next_sequence = np.concatenate([last_sequence[1:], pred_padded], axis=0)
        last_sequence = next_sequence

    # Rescale predictions back to the original scale
    future_predictions = scaler.inverse_transform(
        np.concatenate([np.array(future_predictions).reshape(-1, 1), np.zeros((steps, feature_count - 1))], axis=1)
    )[:, 0]
    return future_predictions

# Download stock data
symbol = input("Stock Symbol (e.g., TSLA, AAPL): ")
print(f"Downloading historical data for {symbol} (5y)...")

try:
    df = yf.download(symbol, period="5y")
    if df.empty:
        raise ValueError(f"No data found for {symbol}. The symbol may be invalid or delisted.")
except Exception as e:
    print(e)
    exit()

# Feature Engineering
df["SMA_20"] = df["Close"].rolling(window=20).mean()  # 20-day Simple Moving Average
df["EMA_10"] = df["Close"].ewm(span=10).mean()       # 10-day Exponential Moving Average
df["RSI"] = compute_rsi(df["Close"])                 # Add Relative Strength Index (RSI)

# Drop NaN values introduced by rolling calculations
df = df.dropna()

# Select features and normalize
features = ["Close", "SMA_20", "EMA_10", "RSI"]
scaler = MinMaxScaler().fit(df[features])
data_scaled = scaler.transform(df[features])

# Prepare data for LSTM
TIMESTEP = 50
X, y = [], []

for i in range(TIMESTEP, len(data_scaled)):
    X.append(data_scaled[i-TIMESTEP:i])  # Last TIMESTEP rows as input
    y.append(data_scaled[i, 0])         # Predict 'Close' price

X, y = np.array(X), np.array(y)

# Split data into training and testing
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(TIMESTEP, len(features))),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

# Train the Model
EPOCHS = 50
BATCH_SIZE = 32
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * (0.95 ** epoch))

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[lr_scheduler],
    verbose=1
)

# Evaluate Model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Predict current test data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(
    np.concatenate([predictions, np.zeros((len(predictions), len(features) - 1))], axis=1)
)[:, 0]
actual = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), len(features) - 1))], axis=1)
)[:, 0]

# Plot Predictions
plt.figure(figsize=(12, 6))
plt.plot(actual, label="Actual Prices", color="blue")
plt.plot(predictions, label="Predicted Prices", color="orange")
plt.title(f"{symbol} Stock Price Prediction (Current Data)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

# Forecast future prices (50 years = ~12500 trading days assuming 250 trading days per year)
FUTURE_DAYS = 12500
future_predictions = forecast_future(data_scaled, model, scaler, FUTURE_DAYS, len(features))

# Plot Future Predictions
plt.figure(figsize=(12, 6))
plt.plot(range(len(df)), scaler.inverse_transform(data_scaled)[:, 0], label="Historical Prices", color="blue")
plt.plot(range(len(df), len(df) + FUTURE_DAYS), future_predictions, label="Future Predictions", color="green")
plt.title(f"{symbol} Stock Price Prediction (Next 50 Years)")
plt.xlabel("Trading Days")
plt.ylabel("Price")
plt.legend()
plt.show()

# Save the Model
model.save(f"{symbol}_stock_model.h5")
print(f"Model saved as {symbol}_stock_model.h5")
