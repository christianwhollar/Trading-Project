import streamlit as st
import pandas as pd

@st.cache
def get_unmatched_blotter():
    return pd.read_csv('outputs/unmatched_blotter.csv')

@st.cache
def get_matched_blotter():
    return pd.read_csv('outputs/matched_blotter.csv')

@st.cache
def get_portfolio_performance():
    df = pd.read_csv('outputs/portfolio_performance.csv')
    df.set_index('Date', inplace=True)
    return df

@st.cache
def get_portfolio_summary():
    return pd.read_csv('outputs/portfolio_summary.csv')

@st.cache
def get_text_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text


col1, col2 = st.columns([8, 1])

with col1:
    st.title('Solana Trading Project')
with col2:
    st.image('./img/solana.png', width = 200)

with col1: 
    st.write("""
        ## Overview
        I developed an MLP model to predict Solana's next day closing price. In short, LSTM models were developed to forecast the future values of various cryptocurrency assets, specifically:
        - Bitcoin (BTC-USD)
        - Ethereum (ETH-USD)
        - Uniswap (UNI-USD)
        - Orca (ORCA-USD)
        - Raydium (RAY-USD)
        An MLP model was trained using the predicted prices and Orca and Raydium TVL data to predict Solana's next day closing price.             
        
        ### Trading Strategy
        The trading strategy operates as follows:
        - If the predicted price exceeds the current price, a position is opened.
        - If the predicted price is below the current price, the position is closed.
        For a comprehensive understanding of decentralized exchanges, including the use of total value locked as an indicator and asset selection criteria for the project, please visit the About page. 
        Detailed explanations of model development are available on the LSTM and MLP pages. 
        Visit the Model Performance page for a detailed breakdown of model results. 
        Interact with the model and predict Solana's price for the next day on the Test page.
    """)

    st.subheader('Performance Metrics')

    st.write("""
        The investment strategy commenced on April 23, 2021, and concluded on March 21, 2024, during which a total of 93 trades were executed. 
        Of these, 58 trades were profitable, yielding an average return of 3.46%, while 35 trades resulted in losses, with an average loss of -2.98%. 
        Overall, the strategy achieved an average return of 1.03% across all trades during the specified period. 
        Please note that the data does not include the number of trading days, as cryptocurrency markets operate continuously. 
        Additionally, Solana gas fees, which are less than $0.01 per transaction, are not included in the calculations due to their negligible amount.
    """)

    # Tabs for displaying various data tables
    performance_tab, unmatched_blotter_tab, matched_blotter_tab, summary_tab = st.tabs([
        "Portfolio Performance", 
        "Unmatched Blotter", 
        "Matched Blotter", 
        "Portfolio Summary"
    ])

    # Data retrieval functions
    unmatched_blotter = get_unmatched_blotter()
    matched_blotter = get_matched_blotter()
    portfolio_performance = get_portfolio_performance()
    portfolio_summary = get_portfolio_summary()

    # Tab contents
    with performance_tab:
        st.dataframe(portfolio_performance)

    with unmatched_blotter_tab:
        st.dataframe(unmatched_blotter)

    with matched_blotter_tab:
        st.dataframe(matched_blotter)

    with summary_tab:
        st.dataframe(portfolio_summary)

    # Line chart for visualizing performance comparison
    st.line_chart(portfolio_performance[['Period Return (%)', 'SPY Period Return (%)']])

    # Call the function to get the code
    text_content = get_text_from_file('./outputs/results_summary.txt')

    # Display the code in your Streamlit app
    st.code(text_content, language='python')

    st.write("""
    #### Calculated alpha (α): 7.599 \n
    #### Calculated beta (β): -1.959         
    """)