# Experiment No. 2A
# Multiclass Classification using Deep Neural Network
# Dataset: UCI Letter Recognition Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop


# ------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"

columns = [
    "letter", "x-box", "y-box", "width", "high", "onpix",
    "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
    "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"
]

data = pd.read_csv(url, names=columns)

print("First 5 rows of dataset:")
print(data.head())

print("\nDataset Shape:", data.shape)

print("\nClass Labels:")
print(data["letter"].unique())


# ------------------------------------------------------
# Step 2: Separate Input and Output
# ------------------------------------------------------

X = data.drop("letter", axis=1)
y = data["letter"]


# ------------------------------------------------------
# Step 3: Encode Target Labels
# ------------------------------------------------------
# A-Z letters are converted into numbers 0-25

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert labels into one-hot encoding
# Example: A -> [1,0,0,...], B -> [0,1,0,...]
y_categorical = to_categorical(y_encoded)


# ------------------------------------------------------
# Step 4: Feature Scaling
# ------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ------------------------------------------------------
# Step 5: Split Dataset
# ------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ------------------------------------------------------
# Step 6: Build Deep Neural Network Model
# ------------------------------------------------------

model = Sequential()

# Input layer + first hidden layer
model.add(Dense(128, activation="relu", input_shape=(16,)))

# Dropout layer to reduce overfitting
model.add(Dropout(0.3))

# Second hidden layer
model.add(Dense(64, activation="relu"))

# Dropout layer
model.add(Dropout(0.2))

# Third hidden layer
model.add(Dense(32, activation="relu"))

# Output layer
# 26 neurons because there are 26 classes A-Z
model.add(Dense(26, activation="softmax"))


# ------------------------------------------------------
# Step 7: Compile Model
# ------------------------------------------------------

model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()


# ------------------------------------------------------
# Step 8: Train Model
# ------------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# ------------------------------------------------------
# Step 9: Evaluate Model
# ------------------------------------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ------------------------------------------------------
# Step 10: Predictions
# ------------------------------------------------------

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nAccuracy Score:", accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=label_encoder.classes_
))


# ------------------------------------------------------
# Step 11: Confusion Matrix
# ------------------------------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 8))
sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title("Confusion Matrix - Letter Recognition")
plt.xlabel("Predicted Letter")
plt.ylabel("Actual Letter")
plt.show()


# ------------------------------------------------------
# Step 12: Accuracy Graph
# ------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# ------------------------------------------------------
# Step 13: Loss Graph
# ------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# ------------------------------------------------------
# Step 14: Test Single Sample Prediction
# ------------------------------------------------------

sample = X_test[0].reshape(1, -1)

prediction = model.predict(sample)
predicted_class_index = np.argmax(prediction)
predicted_letter = label_encoder.inverse_transform([predicted_class_index])

actual_class_index = np.argmax(y_test[0])
actual_letter = label_encoder.inverse_transform([actual_class_index])

print("\nSingle Sample Prediction")
print("Actual Letter:", actual_letter[0])
print("Predicted Letter:", predicted_letter[0])
