


# ── STEP 1: Import Libraries ──────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

print('TensorFlow version:', tf.__version__)

# ── STEP 2: Load Dataset from CSV ────────────────────────────────
df = pd.read_csv('HousingData.csv')
df.columns = df.columns.str.strip().str.upper()

print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('\nMissing values per column:')
print(df.isnull().sum())
df.head()

# ── STEP 3: Fix Missing Values (NaN) ── KEY FIX ──────────────────
# NaN values were causing the model to get stuck at loss=72
# Solution: replace each NaN with the mean of that column
df.fillna(df.mean(numeric_only=True), inplace=True)

print('Missing values AFTER fix:')
print(df.isnull().sum())   # all must be 0
print('\nMEDV stats:')
print(df['MEDV'].describe())

# ── STEP 4: Separate Features and Target ─────────────────────────
X = df.drop(columns=['MEDV']).values
y = df['MEDV'].values

print('X shape:', X.shape)
print('y shape:', y.shape)
print('y sample:', y[:5])
# ── STEP 5: Train-Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print('Train:', X_train.shape, '| Test:', X_test.shape)
# ── STEP 6: Scale Features ────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print('Scaling done.')
# ── STEP 7: Build Deep Neural Network ────────────────────────────
tf.random.set_seed(42)

model = Sequential([
    Input(shape=(13,)),
    Dense(128, activation='relu'),
    Dense(64,  activation='relu'),
    Dense(32,  activation='relu'),
    Dense(1)                        # linear output — no activation
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
model.summary()
# ── STEP 8: Train ─────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

print(f'First 5 losses : {[round(x,2) for x in history.history["loss"][:5]]}')
print(f'Last  5 losses : {[round(x,2) for x in history.history["loss"][-5:]]}')
print(f'\nFinal Train MSE : {history.history["loss"][-1]:.4f}')
print(f'Final Val   MSE : {history.history["val_loss"][-1]:.4f}')
# ── STEP 9: Evaluate on Test Set ─────────────────────────────────
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Test MSE  : {loss:.4f}')
print(f'Test MAE  : {mae:.4f}')
print(f'Test RMSE : {np.sqrt(loss):.4f}')
# ── STEP 10: Predictions ──────────────────────────────────────────
predictions = model.predict(X_test, verbose=0).flatten()

print('First 10 Predictions vs Actual:')
print(f'{"Predicted ($k)":>16}  {"Actual ($k)":>12}  {"Error":>8}')
print('-' * 42)
for i in range(10):
    err = predictions[i] - y_test[i]
    print(f'  {predictions[i]:>12.1f}      {y_test[i]:>8.1f}   {err:>+7.1f}')
  # ── STEP 11: Plots ────────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss (MSE) over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, predictions, alpha=0.6, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', label='Perfect Prediction')
plt.xlabel('Actual Price ($1000s)')
plt.ylabel('Predicted Price ($1000s)')
plt.title('Actual vs Predicted House Prices')
plt.legend()

plt.tight_layout()
plt.show()
  

