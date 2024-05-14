import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('F:\AI Weather Forecast\seattle-weather.csv')

df['date'] = pd.to_datetime(df['date'], errors='coerce')

df = df.dropna(subset=['date', 'temp_max', 'temp_min', 'wind', 'weather'])

df['year'] = df['date'].dt.year
print(df.head())

yearly_temps = df.groupby('year').agg({'temp_max': 'mean', 'temp_min': 'mean'}).reset_index()
# Build figure
model_avg = LinearRegression()
model_avg.fit(yearly_temps[['year']], (yearly_temps['temp_max'] + yearly_temps['temp_min']) / 2)
year_range = range(min(yearly_temps['year']), max(yearly_temps['year']) + 1)
predicted_avg = model_avg.predict(pd.DataFrame(year_range, columns=['year']))
plt.figure(figsize=(12, 6))
plt.plot(yearly_temps['year'], (yearly_temps['temp_max'] + yearly_temps['temp_min']) / 2, marker='o', label='Average Temperature', linestyle='-')
plt.plot(year_range, predicted_avg, marker='', label='Predicted Average Temperature', linestyle='--')
plt.title('Yearly Average Temperatures')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.show()
