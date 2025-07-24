# Stock-price-prediction
SVR-based stock price prediction with time series preprocessing and MinMax scaling. Streamlit interface supports symbol input, evaluation metrics, and rolling average charts
# 📈 Stock Price Prediction with SVR and Streamlit

This project predicts stock prices using **Support Vector Regression (SVR)** and visualizes the results through an interactive **Streamlit** dashboard. Users can input any stock symbol, view actual vs. predicted prices, and explore 100-day and 200-day moving averages.

---

## 🚀 Features

- 🔍 **Dynamic Stock Input**: Enter any valid stock symbol (e.g., GOOG, AAPL, TSLA)
- 📊 **Actual vs. Predicted Visualization**: Compare model predictions with real data
- 📉 **Moving Averages**: View 100-day and 200-day rolling trends
- 📋 **Model Evaluation**: MSE, RMSE, and R² metrics displayed
- 🧠 **SVR Model**: Trained on normalized historical data using MinMaxScaler

---

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
## 🏃‍♂️‍➡️Run the Streamlit app
     streamlit run main.py
  ---
  
## 🛅Requirements
  - streamlit
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - yfinance
  - sklearn

---

## 🧪 Model Details-
  - Algorithm: Support Vector Regression (SVR)
  - Preprocessing: MinMaxScaler normalization
  - Training: 80% of historical data
  - Evaluation: 20% test split with metrics


