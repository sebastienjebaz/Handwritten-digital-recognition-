
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant').reshape(-1,32,32,1) / 255.0
x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant').reshape(-1,32,32,1) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build model
model = Sequential([
    Lambda(lambda x: (x - 0.5) / 0.5, input_shape=(32,32,1)),
    Conv2D(6, (5,5), activation='tanh', padding='same'),
    AveragePooling2D(pool_size=(2, 2)),
    Conv2D(16, (5,5), activation='tanh'),
    AveragePooling2D(pool_size=(2, 2)),
    Conv2D(32, (3,3), activation='relu'),
    Dropout(0.3),
    Flatten(),
    Dense(120, activation='tanh'),
    Dense(84, activation='tanh'),
    Dense(10, activation='softmax')
])

# Compile & train
model.compile(optimizer=SGD(0.05, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

def lr_schedule(epoch):
    return 0.05 * (0.5 if epoch > 10 else 1.0) * (0.1 if epoch > 15 else 1.0)

history = model.fit(
    x_train, y_train, batch_size=128, epochs=20, validation_split=0.2,
    callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule)]
)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# Visualize predictions
preds = model.predict(x_test[:5])
plt.figure(figsize=(10,4))
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i].reshape(32,32), cmap='gray')
    plt.title(f"True: {np.argmax(y_test[i])}")
    plt.axis('off')
    plt.subplot(2,5,i+6)
    plt.bar(range(10), preds[i])
    plt.title(f"Pred: {np.argmax(preds[i])}")
plt.tight_layout()
plt.show()
