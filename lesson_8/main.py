#from fastai import datasets as fastai_datasets
import gzip
import numpy as np
import os
import pdb
from sklearn.model_selection import train_test_split
import struct
import torch
from torch import tensor

import matplotlib.pyplot as plt

def plot_digit(data, true_label, predict_label=None):

    if predict_label is None:
        title = f'label: {true_label}'
    else:
        title = f'label: {true_label}, prediction: {predict_label}'

    plt.imshow(data, cmap='gray')
    plt.title(title)
    plt.show()

def plot_flat_digit(data, true_label, predict_label=None):
    plot_digit(data.view(28,28), true_label)

def load_mnist_images_ubyte(filename):
    # https://medium.com/the-owl/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1

    with gzip.open(filename, 'rb') as f:
        magic = struct.unpack('>4B', f.read(4))

        num_img = struct.unpack('>I',f.read(4))[0]
        num_rows = struct.unpack('>I',f.read(4))[0]
        num_cols = struct.unpack('>I',f.read(4))[0]

        #images_array = np.zeros((num_img,num_rows,num_cols))

        # each pixel = 1 byte
        # 'B' is used since it is of 'unsigned char' C type and 'integer' Python type and has standard size 1 as mentioned in the official documentation of struct.
        # '>' is used since the data is in MSB first (high endian) format used by most non-Intel processors, as mentioned in their original website.
        total_bytes = num_img * num_rows * num_cols
        images_array = np.asarray(struct.unpack('>'+'B'*total_bytes, f.read(total_bytes))).reshape((num_img,num_rows,num_cols))

    return images_array


def load_mnist_labels_ubyte(filename):
    with gzip.open(filename, 'rb') as f:
        magic = struct.unpack('>4B', f.read(4))
        num_labels = struct.unpack('>I',f.read(4))[0]
        total_bytes = num_labels
        labels_array = np.asarray(struct.unpack('>'+'B'*total_bytes, f.read(total_bytes))).reshape((num_labels))

    return labels_array

def mnist_to_flat_images(array_3d):
    return array_3d.reshape(array_3d.shape[0], array_3d.shape[1]*array_3d.shape[2])

def load_mnist():
    # Dataset source http://yann.lecun.com/exdb/mnist/index.html

    train_val_images = load_mnist_images_ubyte('inputs/train-images-idx3-ubyte.gz')
    train_val_labels = load_mnist_labels_ubyte('inputs/train-labels-idx1-ubyte.gz')
    test_images = load_mnist_images_ubyte('inputs/t10k-images-idx3-ubyte.gz')
    test_labels = load_mnist_labels_ubyte('inputs/t10k-labels-idx1-ubyte.gz')
    
    # Transformations.
    #   we one dimensional array for each image, reshaping each 28x28 matrix into 784x1 vectors
    #   numpy arrays are not allowed by Jeremy so dataset are converted into tensors.

    train_val_images = mnist_to_flat_images(train_val_images)
    test_images = mnist_to_flat_images(test_images)

    # Numpy array to tensors
    X_train_val, y_train_val, X_test, y_test = map(tensor, (train_val_images, train_val_labels, test_images, test_labels))

    # Split with the same ratio as the lesson 8: train: 50000, valid: 10000
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=1/6, random_state=42)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def main():
    # -----------------------------------------
    # Prepare Folders and load mnist
    # -----------------------------------------

    for f in ['inputs/', 'outputs/']:
        os.makedirs(f, exist_ok=True)

    # Load mnist into tensors
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_mnist()

    # -----------------------------------------
    # Initial linear model
    # -----------------------------------------
    # y = A * X + B
    weights = torch.randn(784, 10)
    bias = torch.zeros(10)
    plot_flat_digit(X_train[0], y_train[0])
    pdb.set_trace()

    # TODO: Study tensors broadcast!!

    #plot_digit(train_images[0], train_labels[0])
if __name__ == "__main__":
    main()