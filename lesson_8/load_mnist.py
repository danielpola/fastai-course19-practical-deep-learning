import gzip
import matplotlib.pyplot as plt
import numpy as np
import os 
from sklearn.model_selection import train_test_split
import struct
from torch import tensor

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


def load_mnist(inputs_folder):
    """ Function to load the mnist files into torch tensors.
    
    Mnist files come in IDS file format. '(...) is a simple format for vectors and multidimensional matrices of various numerical types.'
    """

    # Dataset source http://yann.lecun.com/exdb/mnist/index.html

    train_val_images = load_mnist_images_ubyte(os.path.join(inputs_folder, 'train-images-idx3-ubyte.gz'))
    train_val_labels = load_mnist_labels_ubyte(os.path.join(inputs_folder, 'train-labels-idx1-ubyte.gz'))
    test_images = load_mnist_images_ubyte(os.path.join(inputs_folder, 't10k-images-idx3-ubyte.gz'))
    test_labels = load_mnist_labels_ubyte(os.path.join(inputs_folder, 't10k-labels-idx1-ubyte.gz'))
    
    # Transformations.
    #   We want one dimensional array for each image, reshaping each 28x28 matrix into 784x1 vectors
    #   NumPy arrays are not allowed by Jeremy so dataset are converted into tensors.

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