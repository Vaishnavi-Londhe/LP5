# ============================================================
# Experiment No. 3A
# Plant Disease Detection System using CNN
# Dataset: Plant Disease / PlantVillage Dataset
# ============================================================

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# STEP 1: Dataset Path
# ============================================================

DATASET_PATH = "dataset"

if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset folder not found!")
    print("Create a folder named 'dataset' in the same location as this Python file.")
    exit()


# ============================================================
# STEP 2: Check Dataset Images
# ============================================================

allowed_extensions = (".bmp", ".gif", ".jpeg", ".jpg", ".png")

image_count = 0

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(allowed_extensions):
            image_count += 1

print("Total images found:", image_count)

if image_count == 0:
    print("\nERROR: No images found inside dataset folders.")
    print("Please add .jpg, .jpeg, .png, .bmp, or .gif images inside each class folder.")
    print("\nCorrect format:")
    print("dataset/Tomato___healthy/image1.jpg")
    print("dataset/Tomato___Late_blight/image1.jpg")
    print("dataset/Potato___Early_blight/image1.jpg")
    exit()


# ============================================================
# STEP 3: Basic Parameters
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15


# ============================================================
# STEP 4: Load Training Dataset
# ============================================================

train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)


# ============================================================
# STEP 5: Load Validation Dataset
# ============================================================

validation_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)


# ============================================================
# STEP 6: Class Names
# ============================================================

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\nClass Names:")
print(class_names)

print("\nNumber of Classes:", num_classes)


# ============================================================
# STEP 7: Improve Dataset Performance
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# ============================================================
# STEP 8: Data Augmentation
# ============================================================

data_augmentation = Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])


# ============================================================
# STEP 9: Build CNN Model
# ============================================================

model = Sequential([
    Rescaling(1.0 / 255, input_shape=(128, 128, 3)),

    data_augmentation,

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(num_classes, activation="softmax")
])


# ============================================================
# STEP 10: Compile Model
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n========== MODEL SUMMARY ==========")
model.summary()


# ============================================================
# STEP 11: Train Model
# ============================================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ============================================================
# STEP 12: Evaluate Model
# ============================================================

loss, accuracy = model.evaluate(validation_dataset)

print("\n========== MODEL EVALUATION ==========")
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)


# ============================================================
# STEP 13: Plot Accuracy Graph
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
# STEP 14: Plot Loss Graph
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
# STEP 15: Save Model
# ============================================================

model.save("plant_disease_cnn_model.h5")

print("\nModel saved successfully as plant_disease_cnn_model.h5")


# ============================================================
# STEP 16: Predict Single Image
# ============================================================

def predict_plant_disease(image_path):
    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return

    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions[0]) * 100

    print("\n========== SINGLE IMAGE PREDICTION ==========")
    print("Image Path:", image_path)
    print("Predicted Disease/Class:", predicted_class)
    print("Confidence:", round(confidence, 2), "%")


# Example:
# predict_plant_disease("test_leaf.jpg")
