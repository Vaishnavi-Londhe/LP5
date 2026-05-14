# ============================================================
# Experiment No. 1
# Boston Housing Price Prediction using Deep Neural Network
# Linear Regression using Deep Learning
# ============================================================

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------------

DATASET_PATH = r"C:\Users\hp\OneDrive\Desktop\LPVI\HPC\DL\boston_test.csv"

df = pd.read_csv(DATASET_PATH)

print("First 5 Records:")
print(df.head())

print("\nColumns in Dataset:")
print(df.columns.tolist())

print("\nDataset Shape:")
print(df.shape)


# ------------------------------------------------------------
# Step 2: Clean Column Names
# ------------------------------------------------------------

df.columns = df.columns.str.strip()

print("\nCleaned Columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# Step 3: Find Target Column Automatically
# ------------------------------------------------------------

possible_targets = ["MEDV", "PRICE", "Price", "price", "target", "TARGET"]

target_col = None

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    print("\nERROR: Target column not found.")
    print("Your dataset does not contain MEDV / PRICE / price / target column.")
    print("\nAvailable columns are:")
    print(df.columns.tolist())
    print("\nSolution:")
    print("1. Use training dataset that contains house price column.")
    print("2. Or add target column in CSV.")
    print("3. If your target column has another name, set it manually in target_col.")
    raise SystemExit

print("\nTarget Column Selected:", target_col)


# ------------------------------------------------------------
# Step 4: Data Cleaning
# ------------------------------------------------------------

df = df.dropna()

df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()


# ------------------------------------------------------------
# Step 5: Split Features and Target
# ------------------------------------------------------------

X = df.drop(target_col, axis=1)
y = df[target_col]


# ------------------------------------------------------------
# Step 6: Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ------------------------------------------------------------
# Step 7: Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------------------------------------------------
# Step 8: Build DNN Model
# ------------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])


# ------------------------------------------------------------
# Step 9: Compile Model
# ------------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

print("\nModel Summary:")
model.summary()


# ------------------------------------------------------------
# Step 10: Train Model
# ------------------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)


# ------------------------------------------------------------
# Step 11: Prediction
# ------------------------------------------------------------

y_pred = model.predict(X_test).flatten()


# ------------------------------------------------------------
# Step 12: Evaluation
# ------------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== Evaluation Metrics ==========")
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)


# ------------------------------------------------------------
# Step 13: Actual vs Predicted Table
# ------------------------------------------------------------

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nSample Predictions:")
print(results.head())


# ------------------------------------------------------------
# Step 14: Plot Loss Graph
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# Step 15: Actual vs Predicted Scatter Plot
# ------------------------------------------------------------

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
