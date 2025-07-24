import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

st.title("Stock Price Prediction with SVR")
stock = st.text_input("Enter Stock Symbol:", value="GOOG")

start = '2011-01-01'
end = '2024-12-12'

if stock:
    data = yf.download(stock, start, end)

    if data.empty:
        st.error("No data found for the given stock symbol.")
    else:
        # Preprocess
        data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
        data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scale = scaler.fit_transform(data_train)

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(
            np.arange(len(data_scale)).reshape(-1, 1),
            data_scale.flatten(),
            test_size=0.2,
            random_state=42
        )
        X_train = x_train.reshape(-1, 1)

        # Train model
        model = SVR()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(x_test)

        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📊 Model Evaluation")
        st.write(f"***Mean Squared Error:*** {mse:.6f}")
        st.write(f"***Root Mean Squared Error:*** {rmse:.6f}")
        st.write(f"***R² Score:*** {r2:.6f}")

        # Actual vs Predicted
        st.subheader("📈 Actual vs Predicted Prices")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(data.index[-len(y_test):], y_test, label='Actual')
        ax1.plot(data.index[-len(y_test):], y_pred, label='Predicted')
        ax1.set_title(f'{stock} Stock Price Prediction')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Normalized Price')
        ax1.legend()
        st.pyplot(fig1)

        # 100-day moving average
        st.subheader("📉 100-Day Moving Average")
        ma100 = data['Close'].rolling(window=100).mean()
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(data['Close'], label='Close Price')
        ax2.plot(ma100, label='100-Day MA')
        ax2.set_title(f'{stock} -100-Day Moving Average')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price')
        ax2.legend()
        st.pyplot(fig2)

        # Visualization: 200-day moving average
        st.subheader("📉 200-Day Moving Average")
        ma200 = data['Close'].rolling(window=200).mean()
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(data['Close'], label='Close Price')
        ax3.plot(ma200, label='200-Day MA')
        ax3.set_title(f'{stock} - 200-Day Moving Average')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Price')
        ax3.legend()
        st.pyplot(fig3)