
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import os

# GPU setup and mixed precision
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU detected.")
else:
    print("❌ GPU not detected. Running on CPU.")

# Load all CIFAR-10 training batches
def load_all_cifar10_batches(base_path):
    x_list, y_list = [], []
    for i in range(1, 6):
        file = os.path.join(base_path, f'data_batch_{i}')
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        data = data.astype('float32') / 255.0
        labels = np.array(batch[b'labels'])
        x_list.append(data)
        y_list.append(labels)
    return np.concatenate(x_list), np.concatenate(y_list)

# Load test batch
def load_cifar10_test_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = data.astype('float32') / 255.0
    labels = np.array(batch[b'labels'])
    return data, labels

# Dataset paths
base_path = 'C:/Masters/Spring2025/ASE/project/cifar-10-python/cifar-10-batches-py'
x_train, y_train = load_all_cifar10_batches(base_path)
x_test, y_test = load_cifar10_test_batch(os.path.join(base_path, 'test_batch'))

# -------------------------------
# 🔁 STEP 1: PRE-TRAIN CNN on CIFAR-10
# -------------------------------
def build_pretrain_model():
    model = keras.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

pretrain_model = build_pretrain_model()
pretrain_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
pretrain_model.save('nas_output/cifar10_pretrained_model.keras')

# -------------------------------
# 🔍 STEP 2: ENAS using pretrained backbone
# -------------------------------
def build_model(hp):
    # Load the pretrained model
    pretrained = keras.models.load_model('nas_output/cifar10_pretrained_model.keras')
    pretrained.trainable = False  # freeze backbone

    model = keras.Sequential()
    model.add(pretrained)

    # NAS search space
    model.add(layers.Dense(
        units=hp.Choice('dense_units', [64, 128]),
        activation='relu'
    ))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('lr', [0.001, 0.0005])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='nas_output',
    project_name='cifar10_from_pretrained'
)

tuner.search(x_train, y_train, epochs=5, validation_split=0.2)

# Train best model
best_model = tuner.get_best_models(num_models=1)[0]
history = best_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
best_model.save('nas_output/best_model_finetuned.keras')

# Evaluate
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=2)
print(f"✅ Final Test Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")
