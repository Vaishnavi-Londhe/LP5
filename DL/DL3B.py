# ============================================================
# Experiment No. 3B
# Convolutional Neural Network using Fashion MNIST Dataset
# Aim: Classify fashion clothing into categories using CNN
# ============================================================

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# STEP 1: Load Fashion MNIST Dataset
# ============================================================

(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()

print("Dataset Loaded Successfully")
print("Training Images Shape:", X_train.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Images Shape:", X_test.shape)
print("Testing Labels Shape:", y_test.shape)


# ============================================================
# STEP 2: Class Names
# ============================================================

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# ============================================================
# STEP 3: Normalize Pixel Values
# ============================================================
# Pixel values are from 0 to 255.
# Convert them into range 0 to 1.

X_train = X_train / 255.0
X_test = X_test / 255.0


# ============================================================
# STEP 4: Reshape Data for CNN
# ============================================================
# CNN requires input shape: height, width, channels.
# Fashion MNIST images are grayscale, so channel = 1.

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

print("\nAfter Reshaping:")
print("Training Images Shape:", X_train.shape)
print("Testing Images Shape:", X_test.shape)


# ============================================================
# STEP 5: Display Sample Images
# ============================================================

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.suptitle("Sample Images from Fashion MNIST Dataset")
plt.show()


# ============================================================
# STEP 6: Build CNN Model
# ============================================================

model = models.Sequential()

# First Convolution + Pooling Layer
model.add(layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu",
    input_shape=(28, 28, 1)
))
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolution + Pooling Layer
model.add(layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation="relu"
))
model.add(layers.MaxPooling2D((2, 2)))

# Third Convolution Layer
model.add(layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation="relu"
))

# Flatten Layer
model.add(layers.Flatten())

# Fully Connected Dense Layer
model.add(layers.Dense(64, activation="relu"))

# Dropout Layer to reduce overfitting
model.add(layers.Dropout(0.3))

# Output Layer
# 10 neurons because Fashion MNIST has 10 classes
model.add(layers.Dense(10, activation="softmax"))


# ============================================================
# STEP 7: Compile Model
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n========== MODEL SUMMARY ==========")
model.summary()


# ============================================================
# STEP 8: Train Model
# ============================================================

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)


# ============================================================
# STEP 9: Evaluate Model
# ============================================================

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n========== MODEL EVALUATION ==========")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# STEP 10: Make Predictions
# ============================================================

predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

print("\nFirst 10 Predictions:")
for i in range(10):
    print(
        "Image", i + 1,
        "| Actual:", class_names[y_test[i]],
        "| Predicted:", class_names[predicted_labels[i]]
    )


# ============================================================
# STEP 11: Plot Accuracy Graph
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# STEP 12: Plot Loss Graph
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training Loss vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# STEP 13: Show Prediction Images
# ============================================================

plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")

    actual = class_names[y_test[i]]
    predicted = class_names[predicted_labels[i]]

    plt.title(f"A: {actual}\nP: {predicted}")
    plt.axis("off")

plt.suptitle("Actual vs Predicted Fashion Categories")
plt.show()


# ============================================================
# STEP 14: Save Model
# ============================================================

model.save("fashion_mnist_cnn_model.h5")

print("\nModel saved successfully as fashion_mnist_cnn_model.h5")
