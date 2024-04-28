import streamlit as st
import pickle

from sklearn.preprocessing import MinMaxScaler
import numpy as np

@st.cache()
def load_model():
    return pickle.load(open('./models/mlp.pkl', 'rb'))

# Min and max values to create MinMaxScalers
bitcoin_max = 76010.30740640535
bitcoin_min = 16904.94213289529

raydium_max = 14.959696597409112
raydium_min = 0.2227526427791131

raydium_tvl_max = 2212157128.383447
raydium_tvl_min = 23226860.015806343

ethereum_max = 5078.191137632552
ethereum_min = 1088.5403229408284

uniswap_max = 44.94013404948299
uniswap_min = 4.201882763100727

# Create a MinMaxScaler for each feature
bitcoin_scaler = MinMaxScaler()
raydium_scaler = MinMaxScaler()
raydium_tvl_scaler = MinMaxScaler()
ethereum_scaler = MinMaxScaler()
uniswap_scaler = MinMaxScaler()

# Fit the MinMaxScaler to the min and max values of the corresponding feature
bitcoin_scaler.fit(np.array([[bitcoin_min], [bitcoin_max]]))
raydium_scaler.fit(np.array([[raydium_min], [raydium_max]]))
raydium_tvl_scaler.fit(np.array([[raydium_tvl_min], [raydium_tvl_max]]))
ethereum_scaler.fit(np.array([[ethereum_min], [ethereum_max]]))
uniswap_scaler.fit(np.array([[uniswap_min], [uniswap_max]]))

# Streamlit page setup
st.title('Solana Price Prediction Tool')
st.write('Enter the predicted prices and TVL to get the predicted Solana price for the day.')

# Input fields for users
bitcoin_predicted_price = st.number_input('Enter the predicted Bitcoin price:', value=65000)
raydium_predicted_price = st.number_input('Enter the predicted Raydium price:', value=1.50)
raydium_tvl = st.number_input('Enter the total value locked (TVL) in Raydium:', value=650000000)
ethereum_predicted_price = st.number_input('Enter the predicted Ethereum price:', value=3200)
uniswap_predicted_price = st.number_input('Enter the predicted Uniswap price:', value=8.00)

# Load Model
mlp_model = load_model()

# Button to make prediction
if st.button('Predict Solana Price'):
    # Scale the input data
    bitcoin_predicted_price_scaled = bitcoin_scaler.transform([[bitcoin_predicted_price]])
    raydium_predicted_price_scaled = raydium_scaler.transform([[raydium_predicted_price]])
    raydium_tvl_scaled = raydium_tvl_scaler.transform([[raydium_tvl]])
    ethereum_predicted_price_scaled = ethereum_scaler.transform([[ethereum_predicted_price]])
    uniswap_predicted_price_scaled = uniswap_scaler.transform([[uniswap_predicted_price]])

    # Reshape the transformed data to match the shape of the input tensor
    x_input_scaled = np.array([bitcoin_predicted_price_scaled, raydium_predicted_price_scaled, raydium_tvl_scaled, ethereum_predicted_price_scaled, uniswap_predicted_price_scaled]).reshape(1, -1)

    # Make a prediction with the MLP model
    prediction = mlp_model.predict(x_input_scaled)[0]
    st.write(f'The predicted Solana price for the day is: ${prediction:.2f}')

# Optional: add explanations or additional sections
st.write('This tool uses machine learning to predict the Solana price based on inputs of predicted prices and TVL from various cryptocurrency assets.')