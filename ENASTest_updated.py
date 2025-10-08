
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import os

# Load all 5 CIFAR-10 training batches
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

# Path to local CIFAR-10 data
base_path = 'C:/Masters/Spring2025/ASE/project/cifar-10-python/cifar-10-batches-py'
x_train, y_train = load_all_cifar10_batches(base_path)

# Load MobileNetV2 backbone
base_model = keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze pretrained layers

# Define model for NAS
def build_model(hp):
    model = keras.Sequential()
    model.add(base_model)
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

# NAS tuner setup
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=5,
    factor=3,
    directory='nas_output',
    project_name='cifar10_nas_pretrained'
)

# Search best hyperparameters
tuner.search(x_train, y_train, epochs=5, validation_split=0.2)

# Train best model
best_model = tuner.get_best_models(num_models=1)[0]
history = best_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save best model
best_model.save('nas_output/best_model_pretrained.keras')

# Load and evaluate on test data
x_test, y_test = load_cifar10_test_batch(os.path.join(base_path, 'test_batch'))
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=2)
print(f"✅ Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# Class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Predict and plot 5 random test images
num_samples = 5
indices = np.random.choice(len(x_test), num_samples, replace=False)
sample_images = x_test[indices]
sample_labels = y_test[indices]
predictions = best_model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(12, 4))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(sample_images[i])
    plt.title(f"Pred: {class_names[predicted_labels[i]]}\nActual: {class_names[sample_labels[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
