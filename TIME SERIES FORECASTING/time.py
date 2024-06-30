import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data
file_path1 = 'Alcohol_Sales.csv'  
file_path2 = 'Miles_Traveled.csv'  

alcohol_sales = pd.read_csv(file_path1)
miles_traveled = pd.read_csv(file_path2)

# Display the first few rows of each dataset
print("Alcohol Sales:")
print(alcohol_sales.head())

print("\nMiles Traveled:")
print(miles_traveled.head())

# Convert the 'DATE' column to datetime format
alcohol_sales['DATE'] = pd.to_datetime(alcohol_sales['DATE'])
miles_traveled['DATE'] = pd.to_datetime(miles_traveled['DATE'])

# Set the 'DATE' column as the index
alcohol_sales.set_index('DATE', inplace=True)
miles_traveled.set_index('DATE', inplace=True)

# Plot the data
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(alcohol_sales, label='Alcohol Sales')
plt.title('Alcohol Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(miles_traveled, label='Miles Traveled', color='orange')
plt.title('Miles Traveled Over Time')
plt.xlabel('Date')
plt.ylabel('Miles')
plt.legend()

plt.tight_layout()
plt.savefig('time_series_data_visualization.png')
print("Plot saved as 'time_series_data_visualization.png'")

# Split the data into training and testing sets
train_alcohol_sales, test_alcohol_sales = train_test_split(alcohol_sales, test_size=0.2, shuffle=False)
train_miles_traveled, test_miles_traveled = train_test_split(miles_traveled, test_size=0.2, shuffle=False)

# Fit the ARIMA model on alcohol sales data
arima_model = ARIMA(train_alcohol_sales, order=(5, 1, 0))
arima_model_fit = arima_model.fit()

# Forecast using ARIMA
arima_forecast = arima_model_fit.forecast(steps=len(test_alcohol_sales))

# Evaluate ARIMA model
arima_rmse = np.sqrt(mean_squared_error(test_alcohol_sales, arima_forecast))
arima_mae = mean_absolute_error(test_alcohol_sales, arima_forecast)

print("ARIMA Model - Alcohol Sales")
print("RMSE:", arima_rmse)
print("MAE:", arima_mae)

# Prepare data for Prophet
alcohol_sales_prophet = alcohol_sales.reset_index().rename(columns={'DATE': 'ds', 'S4248SM144NCEN': 'y'})

# Initialize and fit the Prophet model
prophet_model = Prophet()
prophet_model.fit(alcohol_sales_prophet)

# Make future dataframe and forecast
future = prophet_model.make_future_dataframe(periods=12, freq='M')
prophet_forecast = prophet_model.predict(future)

# Plot forecast
prophet_model.plot(prophet_forecast)
plt.title('Prophet Forecast - Alcohol Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.savefig('prophet_forecast_alcohol_sales.png')
print("Prophet forecast plot saved as 'prophet_forecast_alcohol_sales.png'")

# Evaluate Prophet model on the test set
prophet_forecast_test = prophet_forecast.set_index('ds').loc[test_alcohol_sales.index]
prophet_rmse = np.sqrt(mean_squared_error(test_alcohol_sales, prophet_forecast_test['yhat']))
prophet_mae = mean_absolute_error(test_alcohol_sales, prophet_forecast_test['yhat'])

print("Prophet Model - Alcohol Sales")
print("RMSE:", prophet_rmse)
print("MAE:", prophet_mae)
