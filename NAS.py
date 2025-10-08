import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import numpy as np
import pickle

# # Load CIFAR-10 data
# def load_cifar10_data(data_dir):
#     data = []
#     labels = []
#     for i in range(1, 6):
#         file = f'{data_dir}/data_batch_{i}'
#         with open(file, 'rb') as fo:
#             batch = pickle.load(fo, encoding='bytes')
#             data.append(batch[b'data'])
#             labels.extend(batch[b'labels'])

#     data = np.concatenate(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
#     data = data.astype('float32') / 255.0
#     labels = np.array(labels)
#     return data, labels


# # Load data
# data_dir = 'C:/Masters/Spring2025/ASE/project/cifar-10-python/cifar-10-batches-py'
# x_train, y_train = load_cifar10_data(data_dir)


# Loadind data
def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = data.astype('float32') / 255.0
    labels = np.array(batch[b'labels'])
    return data, labels

file = 'C:/Masters/Spring2025/ASE/project/cifar-10-python/cifar-10-batches-py/data_batch_1'
x_train, y_train = load_cifar10_batch(file)

# Model
def build_model(hp):
    model = keras.Sequential()
    
    # First Conv layer
    model.add(layers.Conv2D(
        filters=hp.Int('filters_1', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
        activation='relu',
        input_shape=(32, 32, 3)
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Additional Conv layers
    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(layers.Conv2D(
            filters=hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5]),
            activation='relu'
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
        activation='relu'
    ))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [0.001, 0.0001])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='nas_output',
    project_name='cifar10_nas'
)

tuner.search(x_train, y_train, epochs=10, validation_split=0.2)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Train the best model
history = best_model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2
)

# Evaluate the model
loss, accuracy = best_model.evaluate(x_train, y_train)
print(f"Final accuracy: {accuracy:.4f}")

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]
