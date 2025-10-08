import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

# Load a single CIFAR-10 batch
def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = data.astype('float32') / 255.0
    labels = np.array(batch[b'labels'])
    return data, labels

file = 'C:/Masters/Spring2025/ASE/project/cifar-10-python/cifar-10-batches-py/data_batch_1'
x_train, y_train = load_cifar10_batch(file)

# Load a pre-trained model (e.g., MobileNetV2)
base_model = keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    include_top=False,  # Exclude final FC layers
    weights='imagenet'
)
base_model.trainable = False  # Freeze pre-trained layers

# Define NAS search space using pre-trained backbone
def build_model(hp):
    model = keras.Sequential()
    model.add(base_model)  # Use pre-trained model as backbone

    # NAS Search Space (only modifies final layers)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(
        units=hp.Choice('dense_units', [64, 128]),
        activation='relu'
    ))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Use Hyperband for faster NAS search
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=5,  # Reduce epochs for faster search
    factor=3,
    directory='nas_output',
    project_name='cifar10_nas_pretrained'
)

# Start NAS search
tuner.search(x_train, y_train, epochs=5, validation_split=0.2)

# Get best model & train
best_model = tuner.get_best_models(num_models=1)[0]
history = best_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# ✅ Save the optimized model
best_model.save('nas_output/best_model_pretrained.keras')
