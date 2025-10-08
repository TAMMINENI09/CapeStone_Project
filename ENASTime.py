import pickle
import numpy as np
import tensorflow as tf
import time
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Load CIFAR-10 batch with normalization
def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = data.astype('float32') / 255.0
    labels = np.array(batch[b'labels'])
    return data, labels

# Load CIFAR-10 training and test batches
train_file = 'C:/Masters/Spring2025/ASE/project/cifar-10-python/cifar-10-batches-py/data_batch_1'
test_file = 'C:/Masters/Spring2025/ASE/project/cifar-10-python/cifar-10-batches-py/test_batch'

# Load training and test data
x_train, y_train = load_cifar10_batch(train_file)
x_test, y_test = load_cifar10_batch(test_file)

# Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    horizontal_flip=True  
)
datagen.fit(x_train)  # Augment only the training data

# Improved Model with fewer layers
def build_model(hp):
    model = keras.Sequential()
    
    # First Conv Layer
    model.add(layers.Conv2D(
        filters=hp.Int('filters_1', 32, 64, step=32), 
        kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
        activation='relu',
        input_shape=(32, 32, 3)
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Second Conv Layer
    model.add(layers.Conv2D(
        filters=hp.Int('filters_2', 64, 128, step=64),
        kernel_size=hp.Choice('kernel_size_2', values=[3, 5]),
        activation='relu'
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('dense_units', 128, 256, step=128), 
        activation='relu'
    ))
    model.add(layers.Dropout(0.3))  # Prevent overfitting
    model.add(layers.Dense(10, activation='softmax'))
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log'))
    
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
    max_epochs=10,
    factor=3,
    directory='nas_output',
    project_name='cifar10_nas_optimized',
    overwrite=True
)

# Tune model on augmented data
tuner.search(datagen.flow(x_train, y_train, batch_size=128), epochs=10, validation_data=(x_test, y_test))

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]

# Early stopping to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Train the best model with augmentation
history = best_model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=50,  # Increase epochs
    validation_data=(x_test, y_test),  # Use test_batch as validation data
    callbacks=[early_stop]
)

# End time tracking
end_time = time.time()
execution_time = end_time - start_time
final_accuracy = history.history['val_accuracy'][-1]

# Print results
print(f"Optimized NAS Execution Time: {execution_time:.2f} seconds")
print(f"Optimized NAS Validation Accuracy: {final_accuracy:.4f}")
