import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('seattle-weather.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date', 'temp_max', 'temp_min', 'wind', 'weather'])
df['year'] = df['date'].dt.year
print(df.head())

yearly_temps = df.groupby('year').agg({'temp_max': 'max', 'temp_min': 'min'}).reset_index()
print(yearly_temps.head())

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_temps, x='year', y='temp_min', label='Min Temperature', marker='o')
sns.lineplot(data=yearly_temps, x='year', y='temp_max', label='Max Temperature', marker='o')
plt.title('Yearly Min and Max Temperatures')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.legend()
plt.show()
