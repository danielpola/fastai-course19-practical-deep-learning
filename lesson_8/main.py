#from fastai import datasets as fastai_datasets
import gzip
import numpy as np
import math
import os
import pdb
from sklearn.model_selection import train_test_split
import struct
import torch
from torch import tensor

import matplotlib.pyplot as plt

def normalize(x, mean, std):
    return (x - mean) / std


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
    return array_3d.reshape(array_3d.shape[0], array_3d.shape[1]*array_3d.shape[2])/255


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
    X_train_val = X_train_val.float()
    X_test = X_test.float()

    # Split with the same ratio as the lesson 8: train: 50000, valid: 10000
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=1/6, random_state=42)


    # Normalize with respecto to train mean and std. We want them to be 0 and 1. (Validation set must be normalized with train data)
    train_mean, train_std = X_train.mean(), X_train.std()
    print("Train mean and std:", train_mean, train_std)

    X_train = normalize(X_train, train_mean, train_std)
    X_valid = normalize(X_valid, train_mean, train_std)
    X_test = normalize(X_test, train_mean, train_std)

    print("Normalized stats", X_train.mean(), X_train.std(), X_valid.mean(), X_valid.std(), X_test.mean(), X_test.std())

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def lin(x, w, b):
    return x@w + b

def relu(x):
    # Try to use single pytorch calls. Fast code implemented in C!
    return x.clamp_min(0.)

def basic_arquitecture(X_train, y_train, X_valid, y_valid):
    # c: number of activations. Normally we would want to use cross-entropy against the 10 activations but to simplify for now
    # we are going to use MSE that means we are gonna have one activation.
    # n: Num of examples (50.000), m: num of columns (784 pixels)
    n, m = X_train.shape
    c = y_train.max() + 1

    # One hidden layer of 50 nodes
    num_hidden = 50

    # One input layer and one hidden layer => 2 weight matrices and 2 bias vectos
    # Simplified kaiming initialization -> dividing by sqrt to get mean of 0 and std of 1
    w1 = torch.randn(m, num_hidden) / math.sqrt(m)
    b1 = torch.zeros(num_hidden)
    w2 = torch.randn(num_hidden, 1) / math.sqrt(num_hidden)
    b2 = torch.zeros(1)

    # First layer is a ReLU, then the output will have a positive mean and a std close to half the original std (because negatives are converted to 0)
    # After a few layers std ->0

    # Kaiming initialization (Read the paper)
    w1 = torch.randn(m, num_hidden) * math.sqrt(2/m)
    b1 = torch.zeros(num_hidden)
    w2 = torch.randn(num_hidden, 1) * math.sqrt(2/num_hidden)
    b2 = torch.zeros(1)
    import pdb
    pdb.set_trace()
    print("input layer stats", relu(lin(X_train, w1, b1)).mean(), relu(lin(X_train, w1, b1)).std())

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
    basic_arquitecture(X_train, y_train, X_valid, y_valid)

    # -----------------------------------------
    # Initial linear model
    # -----------------------------------------
    # y = A * X + B
    weights = torch.randn(784, 10)
    bias = torch.zeros(10)
    plot_flat_digit(X_train[0], y_train[0])
    pdb.set_trace()

    #plot_digit(train_images[0], train_labels[0])
if __name__ == "__main__":
    main()


# TOPICS TO STUDY
# CROSS ENTROPY