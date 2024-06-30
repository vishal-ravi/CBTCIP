# Time Series Analysis and Forecasting

This project demonstrates how to perform time series analysis and forecasting using ARIMA and Prophet models. The datasets used in this project are `Alcohol_Sales.csv` and `Miles_Traveled.csv`. The project includes data visualization, model training, forecasting, and evaluation.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Features](#features)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Usage](#usage)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vishal-ravi/cbtcip.git
   cd time-series-forecasting
   ```

2. Install the required packages:
   ```bash
   pip install numpy pandas matplotlib scikit-learn statsmodels prophet
   ```

## Dataset

The datasets used in this project are stored in CSV files named `Alcohol_Sales.csv` and `Miles_Traveled.csv`. The `Alcohol_Sales.csv` file contains data on alcohol sales over time, and the `Miles_Traveled.csv` file contains data on miles traveled over time. Both datasets have a `DATE` column representing the date of the observation.

## Features

The main feature used for analysis and forecasting is the date, which is used as the time index. The target variable for the `Alcohol_Sales.csv` dataset is the sales data, and for the `Miles_Traveled.csv` dataset, it is the miles traveled data.

## Model Training

### ARIMA Model

The ARIMA (AutoRegressive Integrated Moving Average) model is used for time series forecasting. The model is trained on the `Alcohol_Sales.csv` dataset. The dataset is split into training and testing sets, and the ARIMA model is fitted on the training set.

### Prophet Model

The Prophet model is another time series forecasting model developed by Facebook. The model is also trained on the `Alcohol_Sales.csv` dataset. The dataset is prepared in the format required by Prophet, and the model is fitted on the entire dataset.

## Evaluation

The models' performance is evaluated using the following metrics:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

These metrics are calculated on the testing set to evaluate the ARIMA model and on the forecasted values to evaluate the Prophet model.

## Visualization

The project includes visualizations to understand the data and the forecasting results better:

- **Time Series Data Visualization**: The sales and miles traveled data are plotted over time.
- **ARIMA Forecast**: The forecasted sales values using the ARIMA model.
- **Prophet Forecast**: The forecasted sales values using the Prophet model.

## Usage

To run the project, execute the Python script:

```python
python time.py
```

This will load the datasets, visualize the data, train the ARIMA and Prophet models, make forecasts, evaluate the models, and generate visualizations.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by opening issues or submitting pull requests. For any questions or suggestions, please contact .

---

Happy coding!