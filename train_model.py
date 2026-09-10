import os
os.environ["KERAS_BACKEND"] = "torch"
import shutil
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input

print("Step 1: Downloading data up to today (2026)...")
start = '2012-01-01'
end = date.today().strftime('%Y-%m-%d')
stock = 'GOOG'

data = yf.download(stock, start=start, end=end)
print(f"Total downloaded days: {len(data)}, Date range: {data.index[0].date()} to {data.index[-1].date()}")

close_data = pd.DataFrame(data['Close'].dropna())
training_len = int(len(close_data) * 0.80)
data_train = close_data.iloc[0:training_len]

print(f"Training data points: {len(data_train)}")

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scale = scaler.fit_transform(data_train)

x, y = [], []
for i in range(100, len(data_train_scale)):
    x.append(data_train_scale[i - 100:i])
    y.append(data_train_scale[i, 0])

x, y = np.array(x), np.array(y)
print(f"Features shape: {x.shape}, Target shape: {y.shape}")

print("Step 2: Building LSTM architecture...")
model = Sequential([
    Input(shape=(x.shape[1], 1)),
    LSTM(units=50, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(units=60, activation='relu', return_sequences=True),
    Dropout(0.3),
    LSTM(units=80, activation='relu', return_sequences=True),
    Dropout(0.4),
    LSTM(units=120, activation='relu'),
    Dropout(0.5),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

print("Step 3: Training model for 15 epochs with batch_size=64...")
model.fit(x, y, epochs=15, batch_size=64, verbose=1)

# Step 4: Save model locally and to fallback
local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Stock Predictions Model.keras')
model.save(local_path)
print(f"Model saved to: {local_path}")

fallback_dir = r'C:\Python\Stock'
os.makedirs(fallback_dir, exist_ok=True)
fallback_path = os.path.join(fallback_dir, 'Stock Predictions Model.keras')
shutil.copyfile(local_path, fallback_path)
print(f"Model copied to fallback: {fallback_path}")

print("Model updated to 2026 successfully!")
