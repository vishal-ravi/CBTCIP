import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path1 = 'Unemployment in India.xlsx'  
file_path2 = 'Unemployment_Rate_upto_11_2020.xlsx'  

df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)

# Display the first few rows of each dataset
print("Unemployment in India:")
print(df1.head())

print("\nUnemployment Rate up to 11/2020:")
print(df2.head())

# Check for missing values
print("\nMissing values in Unemployment in India dataset:")
print(df1.isnull().sum())

print("\nMissing values in Unemployment Rate up to 11/2020 dataset:")
print(df2.isnull().sum())

# Drop or fill missing values if any (here, I'm dropping them for simplicity)
df1_cleaned = df1.dropna()
df2_cleaned = df2.dropna()

# Convert the 'Date' column to datetime format
df1_cleaned['Date'] = pd.to_datetime(df1_cleaned['Date'], format='%d-%m-%Y')
df2_cleaned['Date'] = pd.to_datetime(df2_cleaned['Date'], format='%d-%m-%Y')

# Ensure the data types are appropriate
print("\nData types in Unemployment in India dataset:")
print(df1_cleaned.dtypes)

print("\nData types in Unemployment Rate up to 11/2020 dataset:")
print(df2_cleaned.dtypes)

# Set the figure size for better readability
plt.figure(figsize=(14, 7))

# Plot the unemployment rate over time from the first dataset
plt.subplot(1, 2, 1)
sns.lineplot(data=df1_cleaned, x='Date', y='Estimated Unemployment Rate (%)')
plt.title('Unemployment Rate in India Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')

# Plot the unemployment rate over time from the second dataset
plt.subplot(1, 2, 2)
sns.lineplot(data=df2_cleaned, x='Date', y='Estimated Unemployment Rate (%)')
plt.title('Unemployment Rate Up to 11/2020')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')

plt.tight_layout()
plt.show()

# Extract data during the COVID-19 period (e.g., from March 2020 onwards)
covid_period1 = df1_cleaned[df1_cleaned['Date'] >= '2020-03-01']
covid_period2 = df2_cleaned[df2_cleaned['Date'] >= '2020-03-01']

# Calculate the mean unemployment rate before and during COVID-19
mean_unemployment_before_covid1 = df1_cleaned[df1_cleaned['Date'] < '2020-03-01']['Estimated Unemployment Rate (%)'].mean()
mean_unemployment_during_covid1 = covid_period1['Estimated Unemployment Rate (%)'].mean()

mean_unemployment_before_covid2 = df2_cleaned[df2_cleaned['Date'] < '2020-03-01']['Estimated Unemployment Rate (%)'].mean()
mean_unemployment_during_covid2 = covid_period2['Estimated Unemployment Rate (%)'].mean()

print("\nMean Unemployment Rate Before COVID-19 (Dataset 1):", mean_unemployment_before_covid1)
print("Mean Unemployment Rate During COVID-19 (Dataset 1):", mean_unemployment_during_covid1)

print("\nMean Unemployment Rate Before COVID-19 (Dataset 2):", mean_unemployment_before_covid2)
print("Mean Unemployment Rate During COVID-19 (Dataset 2):", mean_unemployment_during_covid2)
