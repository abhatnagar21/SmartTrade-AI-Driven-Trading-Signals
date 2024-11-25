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
    data = data[['Close', 'Volume']]
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

# Calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['Short_EMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['Long_EMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD_Line'] = data['Short_EMA'] - data['Long_EMA']
    data['Signal_Line'] = data['MACD_Line'].ewm(span=signal_window, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD_Line'] - data['Signal_Line']
    return data

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_sd=2):
    data['Moving_Avg'] = data['Close'].rolling(window=window).mean()
    data['Std_Dev'] = data['Close'].rolling(window=window).std()
    data['Upper_Band'] = data['Moving_Avg'] + (data['Std_Dev'] * num_sd)
    data['Lower_Band'] = data['Moving_Avg'] - (data['Std_Dev'] * num_sd)
    return data

# Generate Buy/Sell Signals
def generate_signals(data):
    data['Signal'] = 0
    data.loc[(data['Short_MA'] > data['Long_MA']) & (data['RSI'] < 40), 'Signal'] = 1  # Buy
    data.loc[(data['Short_MA'] < data['Long_MA']) & (data['RSI'] > 60), 'Signal'] = -1  # Sell
    data['Position'] = data['Signal'].diff()
    return data

# Prepare data for machine learning
def prepare_ml_data(data):
    data['Target'] = data['Position'].shift(-5).fillna(0)
    data['Target'] = data['Target'].map({-1: 0, 0: 1, 1: 2})  # Map to classes
    features = ['Short_MA', 'Long_MA', 'RSI', 'MACD_Line', 'Signal_Line', 'Upper_Band', 'Lower_Band', 'Volume']
    X = data[features].dropna()
    y = data.loc[X.index, 'Target']
    return X, y

# Calculate portfolio performance
def calculate_portfolio_performance(data, initial_capital=100000, transaction_cost=0.001):
    data['Position'] = data['Signal'].shift()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Position'] * data['Daily_Return']
    data['Strategy_Return'] -= transaction_cost * abs(data['Position'].diff())
    data['Portfolio_Value'] = initial_capital * (1 + data['Strategy_Return'].fillna(0)).cumprod()
    return data, initial_capital, data['Portfolio_Value'].iloc[-1]

# Plot results
def plot_results(data, initial_capital, final_value):
    plt.figure(figsize=(14, 12))
    plt.subplot(4, 1, 1)
    plt.plot(data['Close'], label='Close Price', alpha=0.5)
    plt.plot(data['Short_MA'], label='Short MA', color='red', alpha=0.75)
    plt.plot(data['Long_MA'], label='Long MA', color='green', alpha=0.75)
    plt.title(f'Portfolio Performance\nInitial: ${initial_capital} | Final: ${final_value:.2f}')
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

# Prepare data for ML
X, y = prepare_ml_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(eval_metric='mlogloss')
}

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"{name} Accuracy:", accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))
    except ValueError as e:
        print(f"Error with {name}: {e}")

# Portfolio performance
data, initial_capital, final_value = calculate_portfolio_performance(data)
plot_results(data, initial_capital, final_value)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
