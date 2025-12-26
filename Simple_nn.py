import os
# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, Input

# 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess the data
# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Build the Neural Network Model
model = models.Sequential([
    # Explicit Input layer to avoid Keras warnings
    Input(shape=(28, 28)),
    
    # Flatten the 28x28 images into a 1D array of 784 pixels
    layers.Flatten(),
    
    # Dense hidden layer
    layers.Dense(128, activation='relu'),
    
    # Dropout layer
    layers.Dropout(0.2),
    
    # Output layer
    layers.Dense(10, activation='softmax')
])

# 4. Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the Model
print("Training Simple Network (10 Epochs)...")
model.fit(x_train, y_train, epochs=10)

# 6. Evaluate the Model
print("\nEvaluating on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(f'\nTest accuracy: {test_acc:.4f}')

# Optional: Prediction check
predictions = model.predict(x_test[:1])
print(f"Prediction: {predictions[0].argmax()}")
print(f"Actual: {y_test[0]}")