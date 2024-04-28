import json
import pickle
from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from scripts.config import FONT_PATH
from scripts.lstm_model import LSTM

def build_dataframe() -> pd.DataFrame:
    """
    Builds a DataFrame with cryptocurrency price data and TVL data.

    Returns:
        pd.DataFrame: The DataFrame with the data.
    """
    # Initialize an empty DataFrame
    df = pd.DataFrame()

    # Define a dictionary with the names and tickers of the cryptocurrencies
    tickers = {
        'bitcoin': 'BTC-USD',
        'solana': 'SOL-USD',
        'raydium': 'RAY-USD',
        'orca': 'ORCA-USD',
        'ethereum': 'ETH-USD',
        'uniswap': 'UNI7083-USD'
    }

    # Loop over the tickers dictionary to download and process price data
    for name, ticker in tickers.items():
        # Download the price data for each cryptocurrency
        raw_price_data = yf.download(ticker, start='2020-01-01', end='2024-03-31', progress=False)

        # Normalize the price data
        scaler = MinMaxScaler()
        normalized_price_data = scaler.fit_transform(raw_price_data['Close'].values.reshape(-1, 1))

        # Define the model parameters
        input_size = 5
        hidden_layer_size = 250
        output_size = 1

        # Load the LSTM model
        file_path = f'models/lstm_{name}.pkl'
        model = LSTM(input_size, hidden_layer_size, output_size)
        model.load_state_dict(torch.load(file_path))
        
        # Create a tensor for the normalized price data
        x_all = torch.from_numpy(normalized_price_data.astype(np.float32)).view(-1, 1)

        # Predict the prices using a sliding window
        y_all_pred = []
        for i in range(5, len(x_all)):
            x = x_all[i-5:i].view(1, 5, 1)
            y_pred = model.test_model(x)
            y_all_pred.append(y_pred.item())

        # Convert the predicted prices to actual prices
        y_all_pred = scaler.inverse_transform(np.array(y_all_pred).reshape(-1, 1))
        y_all = scaler.inverse_transform(normalized_price_data[5:])

        # Add the actual and predicted prices to the DataFrame
        df[name] = raw_price_data['Close']
        df[f'{name}_predicted'] = np.insert(y_all_pred, 0, [[np.nan]]*(len(df) - len(y_all_pred)), axis=0)

    # Read the TVL data from a CSV file
    df_tvl = pd.read_csv('./data/chain-dataset-Solana.csv')

    # Filter the TVL data for 'Raydium' and 'Orca'
    df_tvl = df_tvl[df_tvl['Protocol'].isin(['Raydium', 'Orca'])].T

    # Drop the first row of the TVL data and convert the index to datetime
    df_tvl.drop(df_tvl.index[0], inplace=True)
    df_tvl.index = pd.to_datetime(df_tvl.index, format='%d/%m/%Y')

    # Rename the columns of the TVL data
    df_tvl.columns = ['raydium_tvl', 'orca_tvl']

    # Drop any NaN values in the TVL data
    df_tvl.dropna(inplace=True)

    # Add the TVL data to the DataFrame
    df['raydium_tvl'] = df_tvl['raydium_tvl']
    df['orca_tvl'] = df_tvl['orca_tvl']

    # Select the columns to use in the model and drop any rows with missing values
    model_data = df[['bitcoin_predicted', 'solana', 'raydium_predicted', 'raydium_tvl', 'ethereum_predicted', 'uniswap_predicted']].dropna()
    return model_data

def build_model(model_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Builds a MLP model using the provided data.

    Args:
        model_data (pd.DataFrame): The data to use for model building.

    Returns:
        Tuple[pd.Series, pd.Series]: The actual and predicted values.
    """
    # Split the data into input (X) and output (y) variables, with 'solana' as the target variable
    X = model_data.drop('solana', axis=1)
    y = model_data['solana']

    # Split the data into training and testing sets, with 20% of the data reserved for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data to have a mean of 0 and a standard deviation of 1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the MLP model with two hidden layers of 50 neurons each, a learning rate of 0.001, and a maximum of 1500 iterations
    model = MLPRegressor(hidden_layer_sizes=(50, 50), learning_rate_init=0.001, max_iter=1500, random_state=42)

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Make predictions for the test set
    y_pred = model.predict(X_test)

    # Save the trained model to a .pkl file
    with open('./models/mlp.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Create a dictionary with the metrics
    metrics = {
        "mae": mae,
        "mse": mse,
        "r2": r2
    }

    # Write the metrics to a JSON file
    with open('./metrics/mlp_test_error_metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Transform all the input data using the previously fitted scaler
    X_all = scaler.transform(X)

    # Make predictions for all the data
    y_all_pred = model.predict(X_all)

    # Convert the predictions to a pandas Series with the same index as y
    y_all_pred = pd.Series(y_all_pred.flatten(), index=y.index)

    # Calculate the Mean Absolute Error for all the data
    mae_all = mean_absolute_error(y, y_all_pred)

    # Calculate the Mean Squared Error for all the data
    mse_all = mean_squared_error(y, y_all_pred)

    # Calculate the R^2 Score for all the data
    r2_all = r2_score(y, y_all_pred)

    # Create a dictionary with the metrics
    metrics = {
        "mae": mae_all,
        "mse": mse_all,
        "r2": r2_all
    }

    # Write the metrics to a JSON file
    with open('./metrics/mlp_full_error_metrics.json', 'w') as f:
        json.dump(metrics, f)

    return y, y_all_pred

def simulate_trades(y: pd.Series, y_all_pred: pd.Series) -> pd.DataFrame:
    """
    Simulate trades based on predicted and actual prices.
    """
    # Initialize variables
    capital = 10000  # Starting capital
    in_trade = False
    trades = []
    symbol = 'SOL-USD'  # Replace with your symbol

    # Iterate over the data
    for i in range(1, len(y)):
        # Enter the trade if we predict the price will go up tomorrow and we're not already in a trade
        if y_all_pred[i] > y[i-1] and not in_trade:
            in_trade = True
            entry_price = y[i]
            entry_date = y.index[i]
            quantity = capital / entry_price  # Calculate quantity

        # Exit the trade if we predict the price will go down tomorrow and we're in a trade
        elif y_all_pred[i] < y[i-1] and in_trade:
            in_trade = False
            exit_price = y[i]
            exit_date = y.index[i]
            # Calculate return
            trade_return = np.log(exit_price / entry_price) / (exit_date - entry_date).days * 100
            # Add trade to list
            trades.append([entry_date, symbol, quantity, entry_price, exit_date, exit_price, trade_return, trade_return])
    
    # Create DataFrame
    trades_df = pd.DataFrame(trades, columns=['Entry Date', 'Symbol', 'Quantity', 'Entry Price', 'Exit Date', 'Exit Price', 'Return', 'Return per Trade Day'])
    trades_df['Return (USD)'] = trades_df['Quantity'] * (trades_df['Exit Price'] - trades_df['Entry Price'])
    return trades_df

def save_figure(y: pd.Series, y_all_pred: pd.Series, trades_df: pd.DataFrame) -> None:
    """
    Save the figure of actual vs predicted prices and trade actions.
    """
    font_prop = fm.FontProperties(fname=FONT_PATH, size=16)

    buy_dates = trades_df['Entry Date']
    sell_dates = trades_df['Exit Date']

    y_range = y.max() - y.min()

    slicer_value = 5

    buy_prices_pred = y_all_pred[buy_dates]
    buy_prices_actual = y[buy_dates]
    buy_prices = np.maximum(buy_prices_pred, buy_prices_actual) + y_range * 0.1
    buy_prices = buy_prices[::slicer_value]  # Select every fifth buy price

    sell_prices_pred = y_all_pred[sell_dates]
    sell_prices_actual = y[sell_dates]
    sell_prices = np.maximum(sell_prices_pred, sell_prices_actual) + y_range * 0.1
    sell_prices = sell_prices[::slicer_value]  # Select every fifth sell price

    # Create a date range
    dates = pd.date_range(start='2021-04-14', periods=len(y_all_pred))

    # Plot actual vs predicted prices
    plt.style.use('dark_background')
    plt.figure(figsize=(14,7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=200))
    plt.plot(dates, y, color='cyan', linewidth=2, label=f'actual price')
    plt.plot(dates, y_all_pred, color='magenta', linewidth=2, label=f'predicted price')

    # Plot green and red dots for buy and sell actions
    plt.plot(buy_prices, color='#00FF00', marker='o', linestyle='', markersize=4)  # Bright green dots for buy actions
    plt.plot(sell_prices, 'ro', markersize=4)  # Red dots for sell actions

    plt.title(f'Solana Price Prediction', fontsize=20, color='white', fontproperties=font_prop)
    plt.xlabel('Time', fontsize=16, color='white', fontproperties=font_prop)
    plt.ylabel(f'Solana Price', fontsize=16, color='white', fontproperties=font_prop)
    plt.legend(fontsize=14, loc='upper right', prop=font_prop)
    plt.grid(color='gray', linestyle='dashed', linewidth=0.5)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig(f'./plots/final_price_prediction.png')

    # Calculate the metrics
    trade_metrics = {
        "Number of trades": len(trades_df),
        "Number of profitable trades": len(trades_df[trades_df['Return'] > 0]),
        "Number of losing trades": len(trades_df[trades_df['Return'] < 0]),
        "Return with 10K USD capital per trade": sum(trades_df['Return (USD)']),
        "Average return of profitable trades": trades_df[trades_df['Return'] > 0]['Return'].mean(),
        "Average return of losing trades": trades_df[trades_df['Return'] < 0]['Return'].mean(),
        "Overall average return": trades_df['Return'].mean()
    }

    # Save the metrics to a JSON file
    with open('./metrics/trade_metrics.json', 'w') as f:
        json.dump(trade_metrics, f)

    trades_df.to_csv('./outputs/blotter.csv', index=False)

if __name__ == "__main__":
    model_data = build_dataframe()
    y, y_all_pred = build_model(model_data)
    trades_df = simulate_trades(y, y_all_pred)
    save_figure(y, y_all_pred, trades_df)