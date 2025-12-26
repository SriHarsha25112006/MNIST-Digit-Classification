import os
# 0 = all messages, 1 = no INFO, 2 = no INFO/WARN, 3 = no INFO/WARN/ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras import layers, models, Input

# 1. Load Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess
# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape data to include the "Channel" dimension
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# 3. Build the CNN Model
model = models.Sequential([
    # Explicit Input layer prevents the "UserWarning" you saw earlier
    Input(shape=(28, 28, 1)),
    
    # First Convolution layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Second Convolution layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Convolution layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4. Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train
print("Training CNN (10 Epochs)...")
model.fit(x_train, y_train, epochs=10)

# 6. Evaluate
print("\nEvaluating CNN...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")