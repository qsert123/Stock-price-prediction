#import required lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score

#data download
start = '2011-01-01'
end = '2024-12-12'
stock = 'GOOG'

data =yf.download(stock, start, end)

#preprocess data & scale features

moving_average = data['Close'].rolling(100).mean()

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])
scaler = MinMaxScaler(feature_range=(0, 1))
data_scale = scaler.fit_transform(data_train, data_test)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    np.arange(len(data_scale)).reshape(-1, 1),
    data_scale.flatten(),
    test_size=0.2, random_state=42
)
X_train = x_train.reshape(-1, 1)

#model training
model = SVR()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')

# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test):], y_test, label='Actual')
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted')
plt.title(stock + ' Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.legend()
plt.show()


