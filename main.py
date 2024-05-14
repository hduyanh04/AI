import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def display_dataframe_head(df, num_rows=5):
    """Display the first few rows of the DataFrame."""
    print("First", num_rows, "rows of the DataFrame:")
    print(df.head(num_rows))

# Load data
file_path = r"F:\AI Weather Forecast\seattle-weather.csv"
data = pd.read_csv(file_path)

# One-hot encode the 'weather' column
data = pd.get_dummies(data, columns=['weather'])

# Display a few rows of the DataFrame
display_dataframe_head(data)

# Select features and target variables
features = ['date', 'precipitation', 'temp_max', 'temp_min', 'wind'] 
X = data[features]
y = data[['temp_max', 'temp_min']]

# Extract year from the date
X['year'] = pd.to_datetime(X['date']).dt.year

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract years from the test set
years_test = X_test['year']

# Drop unnecessary columns
X_train = X_train.drop(columns=['date', 'year'])
X_test = X_test.drop(columns=['date', 'year'])

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

# Create DataFrame for actual temperatures
actual_temps = pd.DataFrame({'Year': years_test, 'Actual Temp Max': y_test['temp_max'], 'Actual Temp Min': y_test['temp_min']})

# Create DataFrame for predicted temperatures
predicted_temps = pd.DataFrame({'Year': years_test, 'Predicted Temp Max': predictions[:, 0], 'Predicted Temp Min': predictions[:, 1]})

# Group by year and calculate the mean temperatures
actual_temps_mean = actual_temps.groupby('Year').mean()
predicted_temps_mean = predicted_temps.groupby('Year').mean()

# Plot the actual and predicted temperature max and min over years
plt.figure(figsize=(12, 6))  # Adjust the figure size
plt.plot(actual_temps_mean.index, actual_temps_mean['Actual Temp Max'], label='Actual Temp Max', color='blue', marker='o')  # Actual Temp Max
plt.plot(actual_temps_mean.index, actual_temps_mean['Actual Temp Min'], label='Actual Temp Min', color='red', marker='o')  # Actual Temp Min
plt.plot(predicted_temps_mean.index, predicted_temps_mean['Predicted Temp Max'], label='Predicted Temp Max', color='blue', linestyle='--', marker='x')  # Predicted Temp Max
plt.plot(predicted_temps_mean.index, predicted_temps_mean['Predicted Temp Min'], label='Predicted Temp Min', color='red', linestyle='--', marker='x')  # Predicted Temp Min
plt.xlabel('Year')
plt.ylabel('Temperature (°F)')
plt.title('Average Actual vs Predicted Temperature Over Years')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis ticks
plt.tight_layout()
plt.show()
