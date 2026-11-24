# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create/load the environmental sensor dataset containing humidity, wind speed, rainfall, and other sensor values with daily temperature, PM2.5 level, and energy consumption.
2. Split the dataset into training and testing sets.
3. Train the Random Forest Regressor model using the training data and predict the daily temperature, PM2.5 pollution level, and energy consumption.
4. Evaluate the model and predict the output values for a new environmental sensor reading.

## Program:

~~~py

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

# Sample Dataset
# Features: Humidity(%), Wind Speed(km/h), Rainfall(mm)
X = np.array([
    [65, 10, 5],
    [70, 12, 8],
    [60, 8, 2],
    [75, 15, 10],
    [80, 20, 15],
    [55, 6, 1],
    [68, 11, 4],
    [72, 13, 7],
    [78, 18, 12],
    [58, 7, 3]
])

# Targets: Temperature(°C), PM2.5, Energy(kWh)
Y = np.array([
    [30, 45, 220],
    [29, 50, 230],
    [33, 35, 200],
    [27, 60, 250],
    [25, 75, 280],
    [34, 30, 190],
    [31, 42, 215],
    [28, 55, 240],
    [26, 68, 270],
    [32, 38, 205]
])

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train Random Forest Model
model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, random_state=42)
)

model.fit(X_train, Y_train)

# Prediction
Y_pred = model.predict(X_test)

print("Actual Values:")

print(Y_test)

print("\nPredicted Values:")

print(np.round(Y_pred, 2))

# Accuracy (R2 Score)
score = model.score(X_test, Y_test)

print("\nR2 Score:", round(score, 2))

# Prediction for new sensor data
new_data = np.array([[67, 10, 6]])

prediction = model.predict(new_data)

print("\nPrediction for New Sensor Data")

print("Temperature (°C):", round(prediction[0][0], 2))

print("PM2.5 Level:", round(prediction[0][1], 2))

print("Energy (kWh):", round(prediction[0][2], 2))

~~~
## Output:

<img width="803" height="677" alt="image" src="https://github.com/user-attachments/assets/d2f34d68-f924-4290-825b-5269ac5306da" />


## Result:

Thus, the Random Forest Algorithm to predict the daily temperature, PM2.5 pollution level, and energy consumption based on environmental sensor data was implemented successfully and verified using Python programming.
