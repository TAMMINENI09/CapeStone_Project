import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def show_cifar_images(data, num_images=10):
    images = data[b'data']  # Shape (10000, 3072)
    labels = data[b'labels']
    filenames = [name.decode('utf-8') for name in data[b'filenames']]

    # Reshape the image data to (num_samples, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Convert to (32, 32, 3)

    # Plot the images
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(filenames[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

file = 'C:/Masters/Spring2025/ASE/project/cifar-10-python/cifar-10-batches-py/data_batch_1'
data = unpickle(file)

print(data)

show_cifar_images(data, num_images=10)