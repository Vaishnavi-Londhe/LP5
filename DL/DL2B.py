# Binary Classification using Deep Neural Network
# IMDB Movie Review Sentiment Analysis

# Import Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

# Load IMDB Dataset
# num_words = top 10,000 frequently used words
vocab_size = 10000

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=vocab_size
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# Pad sequences to same length
max_length = 200

X_train = pad_sequences(
    X_train,
    maxlen=max_length,
    padding='post'
)

X_test = pad_sequences(
    X_test,
    maxlen=max_length,
    padding='post'
)

print("Shape after Padding:")
print(X_train.shape)


# Build Deep Neural Network Model
model = Sequential([
    Embedding(input_dim=vocab_size,
              output_dim=32,
              input_length=max_length),

    Flatten(),

    Dense(64, activation='relu'),
    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model Summary
print("\nModel Summary:")
model.summary()


# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# Evaluate Model
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy:", accuracy * 100)


# Predict Sentiment
predictions = model.predict(X_test)

# Convert probability into Positive/Negative
predicted_labels = (
    predictions > 0.5
).astype("int32")

# Show Sample Predictions
print("\nSample Predictions:")

for i in range(10):
    actual = (
        "Positive"
        if y_test[i] == 1
        else "Negative"
    )

    predicted = (
        "Positive"
        if predicted_labels[i] == 1
        else "Negative"
    )

    print(
        f"Review {i+1}: "
        f"Actual = {actual}, "
        f"Predicted = {predicted}"
    )


 # Plot Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)
plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()


# Plot Loss Graph
plt.figure(figsize=(8,5))
plt.plot(
    history.history['loss'],
    label='Training Loss'
)
plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()


# Predict Custom Review
word_index = tf.keras.datasets.imdb.get_word_index()

reverse_word_index = {
    value: key
    for (key, value)
    in word_index.items()
}

def decode_review(text):
    return ' '.join(
        [reverse_word_index.get(i - 3, '?')
         for i in text]
    )

print("\nExample Review:")
print(decode_review(X_test[0]))

print("\nPredicted Sentiment:")
if predicted_labels[0] == 1:
    print("Positive Review")
else:
    print("Negative Review")
  
