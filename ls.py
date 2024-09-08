import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Fetch historical data using yfinance
def get_stock_data(ticker, start='2022-01-01', end='2024-01-01'):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Close', 'Volume']]  # Added Volume
    data.rename(columns={'Close': 'Close'}, inplace=True)
    return data

# Calculate Moving Averages
def calculate_moving_averages(data, short_window, long_window):
    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    return data

# Calculate RSI
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Calculate MACD manually
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['Short_EMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['Long_EMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD_Line'] = data['Short_EMA'] - data['Long_EMA']
    data['Signal_Line'] = data['MACD_Line'].ewm(span=signal_window, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD_Line'] - data['Signal_Line']
    return data

# Calculate Bollinger Bands manually
def calculate_bollinger_bands(data, window=20, num_sd=2):
    data['Moving_Avg'] = data['Close'].rolling(window=window).mean()
    data['Std_Dev'] = data['Close'].rolling(window=window).std()
    data['Upper_Band'] = data['Moving_Avg'] + (data['Std_Dev'] * num_sd)
    data['Lower_Band'] = data['Moving_Avg'] - (data['Std_Dev'] * num_sd)
    return data

# Generate Buy/Sell Signals for Training
def generate_signals(data):
    data['Signal'] = 0
    data['Signal'][(data['Short_MA'] > data['Long_MA']) & (data['RSI'] < 40)] = 1  # Buy Signal
    data['Signal'][(data['Short_MA'] < data['Long_MA']) & (data['RSI'] > 60)] = -1  # Sell Signal
    data['Position'] = data['Signal'].diff()
    return data

# Prepare data for Machine Learning
def prepare_ml_data(data):
    data['Target'] = data['Position'].shift(-5).fillna(0)  # Shift the target further into the future
    
    data['Target'] = data['Target'].map({-1: 0, 0: 1, 1: 2})  # Buy, Hold, Sell
    features = ['Short_MA', 'Long_MA', 'RSI', 'MACD_Line', 'Signal_Line', 'Upper_Band', 'Lower_Band', 'Volume']
    
    X = data[features]
    y = data['Target']
    
    X = X.dropna()
    y = y.loc[X.index]
    
    return X, y

# Calculate portfolio performance
def calculate_portfolio_performance(data, initial_capital=100000, trade_size=1000, transaction_cost=0.001):
    data['Position'] = data['Signal'].shift()
    data['Daily_Return'] = data['Close'].pct_change()
    
    # Calculate strategy returns and transaction costs
    data['Strategy_Return'] = data['Position'] * data['Daily_Return']
    data['Strategy_Return'] -= transaction_cost * abs(data['Position'].diff())
    
    # Update portfolio value
    data['Portfolio_Value'] = initial_capital * (1 + data['Strategy_Return']).cumprod()
    data['Portfolio_Value'].fillna(initial_capital, inplace=True)
    
    final_value = data['Portfolio_Value'].iloc[-1]
    return data, initial_capital, final_value

# Plot results
def plot_results(data, initial_capital, final_value):
    plt.figure(figsize=(14, 12))

    # Plot closing price and moving averages
    plt.subplot(4, 1, 1)
    plt.plot(data['Close'], label='Close Price', color='blue', alpha=0.5)
    plt.plot(data['Short_MA'], label='Short MA', color='red', alpha=0.75)
    plt.plot(data['Long_MA'], label='Long MA', color='green', alpha=0.75)
    plt.plot(data[data['Position'] == 1].index, data['Close'][data['Position'] == 1], '^', markersize=10, color='g', label='Buy Signal')
    plt.plot(data[data['Position'] == -1].index, data['Close'][data['Position'] == -1], 'v', markersize=10, color='r', label='Sell Signal')
    plt.title(f'Price and Moving Averages\nInitial Portfolio Value: ${initial_capital:.2f} | Final Portfolio Value: ${final_value:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()

    # Plot RSI
    plt.subplot(4, 1, 2)
    plt.plot(data['RSI'], label='RSI', color='blue')
    plt.axhline(70, linestyle='--', color='red', alpha=0.5)
    plt.axhline(30, linestyle='--', color='green', alpha=0.5)
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid()

    # Plot MACD
    plt.subplot(4, 1, 3)
    plt.plot(data['MACD_Line'], label='MACD Line', color='blue')
    plt.plot(data['Signal_Line'], label='Signal Line', color='red')
    plt.bar(data.index, data['MACD_Histogram'], label='MACD Histogram', color='gray', alpha=0.3)
    plt.title('MACD')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid()

    # Plot Bollinger Bands
    plt.subplot(4, 1, 4)
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['Upper_Band'], label='Upper Band', color='red')
    plt.plot(data['Lower_Band'], label='Lower Band', color='green')
    plt.title('Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Parameters
ticker = 'AAPL'
short_window = 40
long_window = 100

# Fetch and process data
data = get_stock_data(ticker)
data = calculate_moving_averages(data, short_window, long_window)
data = calculate_rsi(data)
data = calculate_macd(data)
data = calculate_bollinger_bands(data)
data = generate_signals(data)

# Prepare data for machine learning
X, y = prepare_ml_data(data)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train machine learning models
# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
try:
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
except ValueError as e:
    print("Error training Logistic Regression:", e)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
try:
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
except ValueError as e:
    print("Error training Random Forest:", e)

# XGBoost
xgb_model = XGBClassifier(eval_metric='mlogloss')
try:
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
except ValueError as e:
    print("Error training XGBoost:", e)

# Evaluate models
print("Logistic Regression:")
print("Accuracy Score:", accuracy_score(y_test, lr_predictions))
print("Classification Report:\n", classification_report(y_test, lr_predictions))

print("Random Forest:")
print("Accuracy Score:", accuracy_score(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))

print("XGBoost:")
print("Accuracy Score:", accuracy_score(y_test, xgb_predictions))
print("Classification Report:\n", classification_report(y_test, xgb_predictions))

# Calculate portfolio performance
data, initial_capital, final_value = calculate_portfolio_performance(data)

# Plot results
plot_results(data, initial_capital, final_value)
