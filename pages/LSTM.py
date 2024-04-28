import streamlit as st

def get_code_from_file(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    return code

st.title('Long Short-Term Memory Models')

st.write("""
## What is an LSTM Model?
Long Short-Term Memory (LSTM) networks are a subtype of recurrent neural networks (RNN) used for sequential data processing.
          An LSTM layer is composed of four main components:
- Forget Gate
- Input Gate
- Input Node
- Output Gate

### What is the Forget Gate?
In the Forget Gate, both the current short-term memory and the input are each multiplied by corresponding trainable weights, then summed and passed through a sigmoid function.
The output, ranging between 0 and 1, represents the fraction of long-term memory that is retained.

### What is the Input Gate?
Similar to the Forget Gate, the Input Gate processes the current short-term memory and the input by multiplying them with trainable weights, summing the products, and passing the result through a sigmoid function.
The sigmoid output indicates the proportion of new information to be stored in the long-term memory.

### What is the Input Node?
The Input Node, also known as the Cell Node, combines the current short-term memory and the input by multiplying them with trainable weights, summing them, and passing through a tanh function.
The result is a value between -1 and 1, representing potential new information to be added to the long-term memory.

### What is the Output Gate?
The Output Gate involves multiple steps: both the current short-term memory and the input are multiplied by trainable weights, summed, and passed through a sigmoid function. 
Simultaneously, the current long-term memory passes through a tanh function. The combined output of these processes forms the new short-term memory and determines the proportion of it to retain for the next stage.
""")

st.image('./img/LSTM.png', caption='LSTM Model', use_column_width=True)

st.write("""
## LSTM Models Used in This Project
For this project, LSTM models were developed to forecast the future values of various cryptocurrency assets, specifically:
- Bitcoin (BTC-USD)
- Ethereum (ETH-USD)
- Uniswap (UNI-USD)
- Solana (SOL-USD)
- Orca (ORCA-USD)
- Raydium (RAY-USD)

### Model Configuration
Each LSTM model is designed with the following configuration:
- **Input Layer**: Takes the last five closing prices of each cryptocurrency asset to predict future trends.
- **LSTM Layer**: Consists of 250 hidden units, allowing the model to learn patterns in time series data effectively.
- **Output Layer**: A linear layer that maps the output of the LSTM layer to the predicted future value of the asset.

### Training Details
The models were trained to minimize the mean squared error between the predicted and actual values, using the Adam optimization algorithm with a learning rate of 0.001. This setup helps in accurately predicting price movements based on historical data.
""")

# Call the function to get the code
code_content = get_code_from_file('./scripts/lstm_model.py')

# Display the code in your Streamlit app
st.code(code_content, language='python')

st.caption("""
For details on the individual LSTM Model performance, advance to the performance page.
""")