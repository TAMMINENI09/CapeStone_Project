import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

# Load just one batch
def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = data.astype('float32') / 255.0
    labels = np.array(batch[b'labels'])
    return data, labels

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# Load a single batch
file = 'C:/Masters/Spring2025/ASE/project/cifar-10-python/cifar-10-batches-py/data_batch_1'
x_train, y_train = load_cifar10_batch(file)

# Define NAS search space
def build_model(hp):
    model = keras.Sequential()
    
    model.add(layers.Conv2D(
        filters=hp.Int('filters_1', min_value=32, max_value=64, step=32),
        kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
        activation='relu',
        input_shape=(32, 32, 3)
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int('num_conv_layers', 1, 2)):
        model.add(layers.Conv2D(
            filters=hp.Int(f'filters_{i}', min_value=32, max_value=64, step=32),
            kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5]),
            activation='relu'
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=64, max_value=128, step=64),
        activation='relu'
    ))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [0.001, 0.0001])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Use RandomSearch for NAS
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Reduce number of trials
    executions_per_trial=1,
    directory='nas_output',
    project_name='cifar10_nas'
)

# Start search
tuner.search(x_train, y_train, epochs=5, validation_split=0.2)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Train the best model
history = best_model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Save the best model
best_model.save('nas_output/best_model.keras')

# -------------------------------
# ✅ Test on Sample Images
# -------------------------------
# Load saved best model
best_model = keras.models.load_model('nas_output/best_model.keras')

# Pick random images for testing
num_samples = 5
indices = np.random.choice(len(x_train), num_samples, replace=False)
sample_images = x_train[indices]
sample_labels = y_train[indices]

# Get predictions
predictions = best_model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

# Plot images with predictions
plt.figure(figsize=(10, 5))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(sample_images[i])
    plt.title(f"Pred: {class_names[predicted_labels[i]]}\nActual: {class_names[sample_labels[i]]}")
    plt.axis('off')

plt.show()
