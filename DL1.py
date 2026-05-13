# ============================================================
# Experiment No. 1
# Linear Regression using Deep Neural Network
# House Price Prediction using Deep Learning
# ============================================================

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ------------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------------
housing = fetch_california_housing()

# Convert dataset into DataFrame
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["PRICE"] = housing.target

print("First 5 records of dataset:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nChecking missing values:")
print(data.isnull().sum())


# ------------------------------------------------------------
# Step 2: Split Features and Target
# ------------------------------------------------------------
X = data.drop("PRICE", axis=1)
y = data["PRICE"]


# ------------------------------------------------------------
# Step 3: Train-Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ------------------------------------------------------------
# Step 4: Standardization
# ------------------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------------------------------------------------
# Step 5: Build Deep Neural Network Model
# ------------------------------------------------------------
model = Sequential()

model.add(Dense(128, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))

# Output layer for regression problem
model.add(Dense(1))


# ------------------------------------------------------------
# Step 6: Compile Model
# ------------------------------------------------------------
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

print("\nModel Summary:")
model.summary()


# ------------------------------------------------------------
# Step 7: Train Model
# ------------------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# ------------------------------------------------------------
# Step 8: Evaluate Model
# ------------------------------------------------------------
loss, mae = model.evaluate(X_test, y_test, verbose=0)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== Model Evaluation ==========")
print("Mean Squared Error  :", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error :", mae)
print("R2 Score            :", r2)


# ------------------------------------------------------------
# Step 9: Predict New House Price
# ------------------------------------------------------------
# Sample input has 8 features because California Housing dataset has 8 features
# Features:
# MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

new_house = np.array([[8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23]])

new_house_scaled = scaler.transform(new_house)

predicted_price = model.predict(new_house_scaled)

print("\n========== New House Prediction ==========")
print("Predicted House Price:", predicted_price[0][0])


# ------------------------------------------------------------
# Step 10: Plot Training Loss
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend()
plt.show()


# ------------------------------------------------------------
# Step 11: Plot Training MAE
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["mae"], label="Training MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.title("Training MAE vs Validation MAE")
plt.legend()
plt.show()
