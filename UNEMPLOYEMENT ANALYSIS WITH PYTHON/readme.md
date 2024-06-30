# Unemployment Rate Analysis in India

This project analyzes the unemployment rate in India using two datasets: `Unemployment in India.xlsx` and `Unemployment_Rate_upto_11_2020.xlsx`. The analysis includes data cleaning, visualization, and statistical comparison of unemployment rates before and during the COVID-19 pandemic.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Data Cleaning](#data-cleaning)
- [Visualization](#visualization)
- [Statistical Analysis](#statistical-analysis)
- [Usage](#usage)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vishal-ravi/cbtcip.git
   cd unemployment-rate-analysis
   ```

2. Install the required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn openpyxl
   ```

## Dataset

The datasets used in this project are stored in Excel files named `Unemployment in India.xlsx` and `Unemployment_Rate_upto_11_2020.xlsx`. Both datasets contain information on the unemployment rate in India over different time periods.

## Data Cleaning

1. Load the datasets:
   ```python
   import pandas as pd

   file_path1 = 'Unemployment in India.xlsx'
   file_path2 = 'Unemployment_Rate_upto_11_2020.xlsx'

   df1 = pd.read_excel(file_path1)
   df2 = pd.read_excel(file_path2)
   ```

2. Check for and handle missing values:
   ```python
   df1_cleaned = df1.dropna()
   df2_cleaned = df2.dropna()
   ```

3. Convert the `Date` column to datetime format:
   ```python
   df1_cleaned['Date'] = pd.to_datetime(df1_cleaned['Date'], format='%d-%m-%Y')
   df2_cleaned['Date'] = pd.to_datetime(df2_cleaned['Date'], format='%d-%m-%Y')
   ```

## Visualization

The unemployment rates over time are visualized using line plots.

1. Set the figure size:
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns

   plt.figure(figsize=(14, 7))
   ```

2. Plot the unemployment rates:
   ```python
   plt.subplot(1, 2, 1)
   sns.lineplot(data=df1_cleaned, x='Date', y='Estimated Unemployment Rate (%)')
   plt.title('Unemployment Rate in India Over Time')
   plt.xlabel('Date')
   plt.ylabel('Unemployment Rate (%)')

   plt.subplot(1, 2, 2)
   sns.lineplot(data=df2_cleaned, x='Date', y='Estimated Unemployment Rate (%)')
   plt.title('Unemployment Rate Up to 11/2020')
   plt.xlabel('Date')
   plt.ylabel('Unemployment Rate (%)')

   plt.tight_layout()
   plt.savefig('unemployment_rate_analysis.png')
   ```

## Statistical Analysis

The mean unemployment rates before and during the COVID-19 pandemic are calculated and compared.

1. Extract data during the COVID-19 period (e.g., from March 2020 onwards):
   ```python
   covid_period1 = df1_cleaned[df1_cleaned['Date'] >= '2020-03-01']
   covid_period2 = df2_cleaned[df2_cleaned['Date'] >= '2020-03-01']
   ```

2. Calculate the mean unemployment rates:
   ```python
   mean_unemployment_before_covid1 = df1_cleaned[df1_cleaned['Date'] < '2020-03-01']['Estimated Unemployment Rate (%)'].mean()
   mean_unemployment_during_covid1 = covid_period1['Estimated Unemployment Rate (%)'].mean()

   mean_unemployment_before_covid2 = df2_cleaned[df2_cleaned['Date'] < '2020-03-01']['Estimated Unemployment Rate (%)'].mean()
   mean_unemployment_during_covid2 = covid_period2['Estimated Unemployment Rate (%)'].mean()

   print("Mean Unemployment Rate Before COVID-19 (Dataset 1):", mean_unemployment_before_covid1)
   print("Mean Unemployment Rate During COVID-19 (Dataset 1):", mean_unemployment_during_covid1)

   print("Mean Unemployment Rate Before COVID-19 (Dataset 2):", mean_unemployment_before_covid2)
   print("Mean Unemployment Rate During COVID-19 (Dataset 2):", mean_unemployment_during_covid2)
   ```

## Usage

To run the project, execute the Python script:

```python
python un2.py
```

This will load the datasets, clean the data, visualize the unemployment rates, and calculate the mean unemployment rates before and during the COVID-19 pandemic.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by opening issues or submitting pull requests. For any questions or suggestions, please contact [yourname@domain.com](mailto:yourname@domain.com).

---
