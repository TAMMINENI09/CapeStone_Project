import pickle
import numpy as np
import tensorflow as tf
import time
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers

# Load CIFAR-10 batch
def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = data.astype('float32') / 255.0
    labels = np.array(batch[b'labels'])
    return data, labels

# Load all 5 training batches
x_train, y_train = [], []
for i in range(1, 6):
    file = f'C:/Masters/Spring2025/ASE/project/cifar-10-python/cifar-10-batches-py/data_batch_{i}'
    data, labels = load_cifar10_batch(file)
    x_train.append(data)
    y_train.append(labels)

x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

# Data Augmentation Layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# NAS Model (With Augmentation & BatchNormalization)
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Conv2D(
        filters=hp.Int('filters_1', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
        activation=None,
        input_shape=(32, 32, 3)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(
        filters=hp.Int('filters_2', min_value=64, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size_2', values=[3, 5]),
        activation=None
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=128, max_value=512, step=128),
        activation='relu'
    ))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(10, activation='softmax'))

    optimizer = keras.optimizers.Adam(
        learning_rate=hp.Float('learning_rate', min_value=0.0001, max_value=0.01, sampling='log')
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Time tracking
start_time = time.time()

# Use Hyperband for faster NAS
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='nas_output',
    project_name='cifar10_nas_improved_all_batches',
    overwrite=True
)

# Apply data augmentation during training
x_train_augmented = data_augmentation(x_train)

# Search best architecture
tuner.search(x_train_augmented, y_train, epochs=50, validation_split=0.2)

# Final training on best model
best_model = tuner.get_best_models(num_models=1)[0]
history = best_model.fit(x_train_augmented, y_train, epochs=100, batch_size=64, validation_split=0.2)

# Execution time
end_time = time.time()
execution_time = end_time - start_time
final_accuracy = history.history['val_accuracy'][-1]

print(f"Improved NAS (All Batches) Execution Time: {execution_time:.2f} seconds")
print(f"Improved NAS (All Batches) Validation Accuracy: {final_accuracy:.4f}")
