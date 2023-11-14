# Import the required libraries and dependencies
import pandas as pd
import hvplot.pandas
import datetime as dt
import holoviews as hv
from prophet import Prophet
import yfinance as yf
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd

# Define the dataframes
forex_data = {
    'Forex Pair': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CHF', 'USD/CAD', 'NZD/USD'],
    'Yahoo Finance Ticker': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCHF=X', 'USDCAD=X', 'NZDUSD=X']
}

crypto_data = {
    'Cryptocurrency': ['Bitcoin', 'Ethereum', 'Ripple', 'Litecoin'],
    'Yahoo Finance Ticker': ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD']
}

commodity_data = {
    'Commodity': ['Gold', 'Crude Oil (WTI)', 'Silver', 'Natural Gas'],
    'Yahoo Finance Ticker': ['GC=F', 'CL=F', 'SI=F', 'NG=F']
}

indices_data = {
    'Index': ['S&P 500', 'Dow Jones Industrial Average', 'Nasdaq Composite', 'FTSE 100'],
    'Yahoo Finance Ticker': ['^GSPC', '^DJI', '^IXIC', '^FTSE']
}

# Step 1: Set up Streamlit app and user input

# Set page configuration
st.set_page_config(
    page_title="Price Prediction App",
    page_icon="✅",
    layout="wide",
)

st.subheader("Example inputs")

# Sidebar section
st.sidebar.title("Settings")

# Create a select box in the sidebar to choose a dataframe
selected_dataframe = st.sidebar.selectbox("Select a Dataframe", ["Forex", "Cryptocurrency", "Commodities", "Indices"])

# Create a select box for choosing the timeframe interval
selected_interval = st.sidebar.selectbox("Select Timeframe Interval", ["1d", "1wk", "1mo"])

# Create a select box for choosing the ticker symbol
symbol = st.sidebar.text_input('Enter a ticker symbol:', 'BTC-USD')


# Display the selected dataframe in the main content area
if selected_dataframe == "Forex":
    st.write(pd.DataFrame(forex_data))
elif selected_dataframe == "Cryptocurrency":
    st.write(pd.DataFrame(crypto_data))
elif selected_dataframe == "Commodities":
    st.write(pd.DataFrame(commodity_data))
elif selected_dataframe == "Indices":
    st.write(pd.DataFrame(indices_data))




# Step 2: Fetch and prepare the data with the selected timeframe interval
df = yf.download(symbol, period="5y", interval=selected_interval)
df = df[["Close"]]
df = df.reset_index()
df.columns = ["ds", "y"]
df = df.sort_values(by=["ds"], ascending=True)

# Set the frequency of the DateTimeIndex
df["ds"] = pd.to_datetime(df["ds"])
df = df.set_index(pd.DatetimeIndex(df["ds"]))

# Step 3: Fit the Prophet model
model = Prophet()
model.fit(df)

# Step 4: Make predictions
future_trends = model.make_future_dataframe(periods=1000, freq="H")
forecast_trends = model.predict(future_trends)

# Step 5: Visualize the predictions
st.write(f'Price data for {symbol}')
st.line_chart(df.set_index('ds')['y'])

st.write(f'Predicted price data for {symbol}')
fig1 = model.plot(forecast_trends)
st.pyplot(fig1)

st.write(f'Forecast components for {symbol}')
fig2 = model.plot_components(forecast_trends)
st.pyplot(fig2)

forecast_trends = forecast_trends.set_index(["ds"])

# Step 6: Additional analysis

# 1. Performance Metrics
y_true = df['y']
y_pred = forecast_trends.loc[df['ds'], 'yhat']

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = sqrt(mse)

st.write('Mean Absolute Error (MAE):', mae)
st.write('Mean Squared Error (MSE):', mse)
st.write('Root Mean Squared Error (RMSE):', rmse)

# 2. Volatility Analysis
df['returns'] = df['y'].pct_change()
df['volatility'] = df['returns'].rolling(window=20).std()

st.write('Volatility of {}'.format(symbol))
st.line_chart(df['volatility'])

# 3. Moving Averages
df['sma_50'] = df['y'].rolling(window=50).mean()
df['sma_200'] = df['y'].rolling(window=200).mean()

st.write('Moving Averages for {}'.format(symbol))
st.line_chart(df[['y', 'sma_50', 'sma_200']])

# 4. Seasonal Decomposition
result = seasonal_decompose(df['y'], model='multiplicative', period=365)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10))

result.observed.plot(ax=ax1)
ax1.set_ylabel('Observed')
result.trend.plot(ax=ax2)
ax2.set_ylabel('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_ylabel('Residual')

st.write(f'Seasonal Decomposition for {symbol}')
st.pyplot(fig)