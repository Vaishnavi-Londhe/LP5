# Boston Housing Price Prediction using Deep Neural Network

# Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("HousingData.csv")

# Display dataset info
print(df.head())
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values if any
df = df.dropna()

# Convert all columns to numeric (safety)
df = df.apply(pd.to_numeric, errors='coerce')

# Remove any remaining NaN rows
df = df.dropna()

# Check columns
print("\nColumns:")
print(df.columns)

# Target column
target_col = "MEDV"   # Change if needed

# Features and Target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Deep Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu',
                           input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Prediction
y_pred = model.predict(X_test)

# Remove NaN predictions if any
mask = ~np.isnan(y_pred.flatten())

y_test_clean = y_test.iloc[mask]
y_pred_clean = y_pred.flatten()[mask]


# Evaluation
mae = mean_absolute_error(y_test_clean, y_pred_clean)
mse = mean_squared_error(y_test_clean, y_pred_clean)
rmse = np.sqrt(mse)

print("\nEvaluation Metrics:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Actual vs Predicted
results = pd.DataFrame({
    "Actual Price": y_test_clean.values,
    "Predicted Price": y_pred_clean
})

print("\nSample Predictions:")
print(results.head())


# Plot Loss Graph
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# Scatter Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test_clean, y_pred_clean)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()
