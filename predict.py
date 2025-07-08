import numpy as np
from sklearn.linear_model import LinearRegression

# Sample dummy training data (pollutant levels and corresponding AQI values)
# Format: [PM2.5, PM10, NO2, SO2, CO, O3]
X_train = np.array([
    [35, 70, 25, 10, 0.7, 40],
    [80, 150, 60, 20, 1.5, 90],
    [20, 40, 15, 5, 0.3, 20],
    [120, 200, 80, 30, 2.0, 100]
])

y_train = np.array([85, 170, 50, 210])  # Corresponding AQI values

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get user input
print("Enter the following pollutant levels:")
pm25 = float(input("PM2.5: "))
pm10 = float(input("PM10: "))
no2 = float(input("NO2: "))
so2 = float(input("SO2: "))
co = float(input("CO: "))
o3 = float(input("O3: "))

# Predict AQI
input_features = np.array([[pm25, pm10, no2, so2, co, o3]])
predicted_aqi = model.predict(input_features)

print(f"\nPredicted AQI: {predicted_aqi[0]:.2f}")

