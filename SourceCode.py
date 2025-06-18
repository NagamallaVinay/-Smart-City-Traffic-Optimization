import pandas as pd
import numpy as np

# Generation of realistic fake traffic data
data = {
    'timestamp': pd.date_range(start="2023-01-01", periods=500, freq="H"),
    'latitude': np.random.uniform(12.85, 13.15, 500),  # Bangalore coordinates
    'longitude': np.random.uniform(77.45, 77.75, 500),
    'vehicle_count': np.random.randint(20, 300, 500),
    'avg_speed': np.random.uniform(5, 60, 500),
    'congestion_level': np.random.randint(0, 100, 500)
}

df = pd.DataFrame(data)
df.to_csv("traffic_data.csv", index=False)
print("Sample dataset created: traffic_data.csv")



import pandas as pd
df = pd.read_csv("traffic_data.csv")
print(df.head())  # Show first 5 rows
print("\nData columns:", df.columns.tolist())



import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load your generated data
df = pd.read_csv("traffic_data.csv")


# Convert timestamp to hour-of-day
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

# See worst congestion hours
worst_hours = df.groupby('hour')['congestion_level'].mean().sort_values(ascending=False)
print("Worst Traffic Hours:\n", worst_hours.head(3))


conn = sqlite3.connect("traffic.db")
df.to_sql("traffic_data", conn, if_exists="replace", index=False)

# Example SQL query: Find high-congestion locations
high_congestion = pd.read_sql("""
    SELECT latitude, longitude, AVG(congestion_level) as avg_congestion
    FROM traffic_data
    GROUP BY latitude, longitude
    HAVING avg_congestion > 70
    ORDER BY avg_congestion DESC
""", conn)
print("\nHigh Congestion Areas:\n", high_congestion.head())


# Prepare features
X = df[['vehicle_count', 'avg_speed', 'hour']]
y = df['congestion_level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)
print("\nModel Accuracy:", model.score(X_test, y_test))


# Predict for new scenario: 200 cars, speed=25km/h at 5PM
prediction = model.predict([[200, 25, 17]])[0]
print(f"\nPredicted Congestion at 5PM: {prediction:.1f}/100")


# Plot congestion by hour
plt.figure(figsize=(10,4))
df.groupby('hour')['congestion_level'].mean().plot(kind='bar', color='orange')
plt.title("Average Traffic Congestion by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Congestion Level (0-100)")
plt.show()



# Add interactive map 
!pip install folium
import folium

# Create base map
map = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)

# Add congestion markers
for _, row in df.sample(50).iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['congestion_level']/20,
        color='red' if row['congestion_level'] > 50 else 'green',
        popup=f"Congestion: {row['congestion_level']}%"
    ).add_to(map)

map