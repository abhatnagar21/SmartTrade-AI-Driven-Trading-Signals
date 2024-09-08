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

# Fetch historical stock data using yfinance
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data[['Close']]  # Use only the Close price for simplicity
        if data.empty:
            raise ValueError("No data fetched. Check ticker symbol or date range.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

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
    rs = gain / loss.replace(0, np.nan)  # Avoid division by zero
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
    data['Signal'][(data['Short_MA'] > data['Long_MA']) & (data['RSI'] < 30)] = 1  # Buy Signal
    data['Signal'][(data['Short_MA'] < data['Long_MA']) & (data['RSI'] > 70)] = -1  # Sell Signal
    data['Position'] = data['Signal'].diff()
    return data

# Prepare data for Machine Learning
def prepare_ml_data(data):
    data = data.dropna()  # Drop rows with NaN values
    
    # Create target variable
    data['Target'] = data['Position'].shift(-1).fillna(0)
    
    # Map the target variable to positive integers
    data['Target'] = data['Target'].map({-1: 0, 0: 1, 1: 2})
    
    # Features and target
    features = ['Short_MA', 'Long_MA', 'RSI', 'MACD_Line', 'Signal_Line', 'Upper_Band', 'Lower_Band']
    X = data[features]
    y = data['Target']
    
    return X, y

# Calculate portfolio performance
def calculate_portfolio_performance(data, initial_capital=100000):
    data['Position'] = data['Signal'].shift(1).fillna(0)
    data['Daily_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Daily_Return'] * data['Position']
    data['Portfolio_Value'] = initial_capital * (1 + data['Strategy_Return']).cumprod()
    final_value = data['Portfolio_Value'].iloc[-1]
    return data, initial_capital, final_value

# Plot results
def plot_results(data, initial_capital, final_value):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.6)
    plt.plot(data.index, data['Short_MA'], label='Short Moving Average', color='orange', alpha=0.6)
    plt.plot(data.index, data['Long_MA'], label='Long Moving Average', color='red', alpha=0.6)
    plt.plot(data.index, data['Upper_Band'], label='Upper Bollinger Band', color='green', linestyle='--')
    plt.plot(data.index, data['Lower_Band'], label='Lower Bollinger Band', color='green', linestyle='--')
    plt.scatter(data.index, data['Close'], c=data['Signal'], cmap='coolwarm', label='Trading Signals', marker='o', alpha=0.7)
    plt.title(f'Trading Strategy Performance\nInitial Capital: ${initial_capital}, Final Portfolio Value: ${final_value:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
if __name__ == "__main__":
    # Parameters
    ticker = 'AAPL'
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    short_window = 40
    long_window = 100

    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)
    if data.empty:
        exit()

    # Calculate indicators
    data = calculate_moving_averages(data, short_window, long_window)
    data = calculate_rsi(data)
    data = calculate_macd(data)
    data = calculate_bollinger_bands(data)

    # Generate signals
    data = generate_signals(data)

    # Prepare ML data
    X, y = prepare_ml_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)

    # Print model performance
    print("Random Forest:")
    print("Accuracy Score:", accuracy_score(y_test, rf_predictions))
    print("Classification Report:\n", classification_report(y_test, rf_predictions))

    print("Logistic Regression:")
    print("Accuracy Score:", accuracy_score(y_test, lr_predictions))
    print("Classification Report:\n", classification_report(y_test, lr_predictions))

    print("XGBoost:")
    print("Accuracy Score:", accuracy_score(y_test, xgb_predictions))
    print("Classification Report:\n", classification_report(y_test, xgb_predictions))

    # Calculate portfolio performance
    data, initial_capital, final_value = calculate_portfolio_performance(data)

    # Plot results
    plot_results(data, initial_capital, final_value)
