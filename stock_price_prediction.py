import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Fetch Historical Stock Data
ticker = 'AAPL'  # Example: Apple Inc.
data = yf.download(ticker, start='2015-01-01', end='2024-12-31')
data = data[['Close']]
data.dropna(inplace=True)

# Step 2: Prepare the Data
data['Prediction'] = data['Close'].shift(-30)  # Predict 30 days into the future

# Features and labels
X = np.array(data[['Close']][:-30])
y = np.array(data['Prediction'][:-30])

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict and Evaluate
predictions = model.predict(X_test)

# Plotting results
plt.figure(figsize=(10,6))
plt.plot(y_test, label='True Price')
plt.plot(predictions, label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

# Evaluate model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse:.2f}')
