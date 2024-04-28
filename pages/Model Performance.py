import streamlit as st
import json

def get_metrics(name:str = 'bitcoin'):
    with open(f'./metrics/{name}_error_metrics.json', 'r') as file:
        metrics = json.load(file)
    return metrics

st.title('LSTM Model Performance')

st.header('Bitcoin Price Prediction')
st.image('./plots/bitcoin_price_prediction.png')

bitcoin_metrics = get_metrics(name = 'bitcoin')
bitcoin_mae = str(round(bitcoin_metrics['mae']))
bitcoin_mse = str(round(bitcoin_metrics['mse']))
bitcoin_rmse = str(round(bitcoin_metrics['rmse']))

st.write(f"""
### Bitcoin Metrics
Mean Absolute Error: {bitcoin_mae} USD
""")

st.write(f"""
Mean Squared Error: {bitcoin_mse} USD^2
""")

st.write(f"""
Root Mean Squared Error: {bitcoin_rmse} USD
""")

st.header('Ethereum Price Prediction')
st.image('./plots/ethereum_price_prediction.png')

ethereum_metrics = get_metrics(name = 'ethereum')
ethereum_mae = str(round(ethereum_metrics['mae']))
ethereum_mse = str(round(ethereum_metrics['mse']))
ethereum_rmse = str(round(ethereum_metrics['rmse']))

st.write(f"""
### Ethereum Metrics
Mean Absolute Error: {ethereum_mae} USD
""")

st.write(f"""
Mean Squared Error: {ethereum_mse} USD^2
""")

st.write(f"""
Root Mean Squared Error: {ethereum_rmse} USD
""")

st.header('Uniswap Price Prediction')
st.image('./plots/uniswap_price_prediction.png')

uniswap_metrics = get_metrics(name = 'uniswap')
uniswap_mae = str(round(uniswap_metrics['mae']))
uniswap_mse = str(round(uniswap_metrics['mse']))
uniswap_rmse = str(round(uniswap_metrics['rmse']))

st.write(f"""
### Uniswap Metrics
Mean Absolute Error: {uniswap_mae} USD
""")

st.write(f"""
Mean Squared Error: {uniswap_mse} USD^2
""")

st.write(f"""
Root Mean Squared Error: {uniswap_rmse} USD
""")

st.header('Raydium Price Prediction')
st.image('./plots/raydium_price_prediction.png')

raydium_metrics = get_metrics(name = 'raydium')
raydium_mae = str(round(raydium_metrics['mae']))
raydium_mse = str(round(raydium_metrics['mse']))
raydium_rmse = str(round(raydium_metrics['rmse']))

st.write(f"""
### Raydium Metrics
Mean Absolute Error: {raydium_mae} USD
""")

st.write(f"""
Mean Squared Error: {raydium_mse} USD^2
""")

st.write(f"""
Root Mean Squared Error: {raydium_rmse} USD
""")

st.title('MLP Model Performance')

st.header('Solana Price Prediction')
st.image('./plots/final_price_prediction.png')
st.caption('~20% of the trades are shown. Green indicates entry. Red indicates exit.')
solana_metrics = get_metrics(name = 'mlp_test')

solana_mae = str(round(solana_metrics['mae']))
solana_mse = str(round(solana_metrics['mse']))
solana_rmse = str(round(solana_metrics['r2']))

st.write(f"""
### Solana Metrics
Mean Absolute Error: {solana_mae} USD
""")

st.write(f"""
Mean Squared Error: {solana_mse} USD^2
""")

st.write(f"""
Root Mean Squared Error: {solana_rmse} USD
""")
