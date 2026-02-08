import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
ticker = 'TSLA'
start_date = '2024-01-01'
end_date = '2025-05-01'
print(f"Downloading data for {ticker}...")
data = yf.download(ticker, start=start_date, end=end_date)
data.dropna(inplace=True)

# Step 2: Feature Engineering

print("Generating features...")
data['Close_Lag1'] = data['Close'].shift(1)
data['Return_1D'] = data['Close'].pct_change()
data['SMA_10'] = ta.trend.sma_indicator(data['Close'].squeeze())
data['RSI'] = ta.momentum.rsi(data['Close'].squeeze())
data['MACD'] = ta.trend.macd_diff(data['Close'].squeeze())
data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Step 3: Train/Test Split

features = ['Close_Lag1', 'Return_1D', 'SMA_10', 'RSI', 'MACD']
X = data[features]
y = data['Target']
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Step 4: Train Model

print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate Model

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f"\nModel Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Step 6: Plot Actual vs Predicted

plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label='Actual', color='black')
plt.plot(y_test.index, predictions, label='Predicted', color='blue')
plt.title(f'{ticker} - Actual vs Predicted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# Step 7: Strategy Simulation

print("\nSimulating strategy...")
strategy_df = data.iloc[split_index:].copy()
strategy_df['Predicted_Close'] = predictions
strategy_df['Signal'] = (strategy_df['Predicted_Close'] > strategy_df['Close'].squeeze()).astype(int)
strategy_df['Actual_Return'] = strategy_df['Close'].squeeze().pct_change()
strategy_df['Strategy_Return'] = strategy_df['Actual_Return'] * strategy_df['Signal'].shift(1)
strategy_df.dropna(inplace=True)
strategy_df['Cumulative_Market_Returns'] = (1 + strategy_df['Actual_Return']).cumprod()
strategy_df['Cumulative_Strategy_Returns'] = (1 + strategy_df['Strategy_Return']).cumprod()
plt.figure(figsize=(14, 6))
plt.plot(strategy_df.index, strategy_df['Cumulative_Market_Returns'], label='Market Returns', color='gray')
plt.plot(strategy_df.index, strategy_df['Cumulative_Strategy_Returns'], label='Strategy Returns', color='green')
plt.title(f'{ticker} - Strategy vs Market Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Growth of $1')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# Step 8: Predict Future Prices

print("\nPredicting future stock prices...")
future_days = 7
future_predictions = []
recent_data = data.copy()
for _ in range(future_days):
    last_row = recent_data.iloc[-1]

#    input_features = np.array([
#        last_row['Close_Lag1'],
#        last_row['Return_1D'],
#        last_row['SMA_10'],
#        last_row['RSI'],
#        last_row['MACD']
#    ]).reshape(1, -1)

    feature_list = ['Close_Lag1', 'Return_1D', 'SMA_10', 'RSI', 'MACD']
    input_features = np.array(last_row[feature_list]).reshape(1, -1)
    next_pred_close = model.predict(input_features)[0]
    future_predictions.append(next_pred_close)
    new_row = {
        'Close': next_pred_close,
        'Close_Lag1': last_row['Close'],
        'Return_1D': (next_pred_close - last_row['Close']) / last_row['Close'],
        'SMA_10': np.nan,
        'RSI': np.nan,
        'MACD': np.nan,
        'Target': np.nan  
    }
    next_index = recent_data.index[-1] + pd.Timedelta(days=1)
    recent_data = pd.concat([recent_data, pd.DataFrame([new_row], index=[next_index])])
    recent_data['SMA_10'] = ta.trend.sma_indicator(recent_data['Close'])
    recent_data['RSI'] = ta.momentum.rsi(recent_data['Close'])
    recent_data['MACD'] = ta.trend.macd_diff(recent_data['Close'])
recent_actual = data['Close'].tail(60)
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')
future_df = pd.DataFrame({'Predicted_Close': future_predictions}, index=future_dates)

# Step 9: Plot Future Forecast

plt.figure(figsize=(14, 6))
plt.plot(recent_actual.index, recent_actual.values, label='Actual Close (Last 60 Days)', color='black')
plt.plot(future_df.index, future_df['Predicted_Close'], label=f'Predicted Close (Next {future_days} Days)', color='red', linestyle='--')
plt.title(f'{ticker} - Recent 60 Days + Future {future_days} Days Forecast')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
