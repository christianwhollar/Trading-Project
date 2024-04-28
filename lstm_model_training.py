import json

import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import yfinance as yf

from scripts.config import FONT_PATH
from lstm_model import LSTM
from scripts.utils import get_train_test_sets

def train_lstm_models():
    # Load the custom font
    font_prop = fm.FontProperties(fname=FONT_PATH, size=16)

    # Ticker List
    tickers = {
        'bitcoin': 'BTC-USD',
        'solana': 'SOL-USD',
        'raydium': 'RAY-USD',
        'orca': 'ORCA-USD',
        'ethereum': 'ETH-USD',
        'uniswap': 'UNI7083-USD'
        }

    for name, ticker in tickers.items():
        # Download data
        raw_price_data = yf.download(ticker, start='2020-01-01', end='2024-03-31', progress=False)

        # Normalize data
        scaler = MinMaxScaler()
        normalized_price_data = scaler.fit_transform(raw_price_data['Close'].values.reshape(-1, 1))

        # Split data
        x_train, x_test, y_train, y_test = get_train_test_sets(normalized_price_data)

        # Initialize the model, define loss function and optimizer
        model = LSTM()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        model.train_model(x_train, y_train, loss_function, optimizer, epochs=150)

        # Test the model
        y_test_pred = model.test_model(x_test)
        model.save_model(f'./models/lstm_{name}.pkl')

        # Inverse transform to get actual prices
        y_test_pred = scaler.inverse_transform(y_test_pred)
        y_test = scaler.inverse_transform(y_test)

        # Calculate Mean Absolute Error
        mae = mean_absolute_error(y_test, y_test_pred)

        # Calculate Mean Squared Error
        mse = mean_squared_error(y_test, y_test_pred)

        # Calculate Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Create a dictionary to hold the error metrics
        error_metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }

        # Write the dictionary to a JSON file
        with open(f'./metrics/{name}_error_metrics.json', 'w') as f:
            json.dump(error_metrics, f)

        # Create a tensor for the entire data
        x_all = torch.from_numpy(normalized_price_data.astype(np.float32)).view(-1, 1)

        # Predict the prices using a sliding window
        y_all_pred = []
        for i in range(5, len(x_all)):
            x = x_all[i-5:i].view(1, 5, 1)
            y_pred = model.test_model(x)
            y_all_pred.append(y_pred.item())

        # Convert to numpy array and reshape
        y_all_pred = np.array(y_all_pred).reshape(-1, 1)

        # Inverse transform to get actual prices
        y_all_pred = scaler.inverse_transform(y_all_pred)
        y_all = scaler.inverse_transform(normalized_price_data[5:])

        # Create a date range
        dates = pd.date_range(start='2020-01-01', periods=len(y_all))

        # Plot actual vs predicted prices
        plt.style.use('dark_background')
        plt.figure(figsize=(14,7))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=200))
        plt.plot(dates, y_all, color='cyan', linewidth=2, label=f'actual {name} price')
        plt.plot(dates, y_all_pred, color='magenta', linewidth=2, label=f'predicted {name} price')
        plt.title(f'{name} price Prediction', fontsize=20, color='white', fontproperties=font_prop)
        plt.xlabel('time', fontsize=16, color='white', fontproperties=font_prop)
        plt.ylabel(f'{name} price', fontsize=16, color='white', fontproperties=font_prop)
        plt.legend(fontsize=14, loc='lower right', prop=font_prop)
        plt.grid(color='gray', linestyle='dashed', linewidth=0.5)
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.savefig(f'./plots/{name}_price_prediction.png')

if __name__ == "__main__":
    train_lstm_models()