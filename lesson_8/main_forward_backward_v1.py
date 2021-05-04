import numpy as np
import math
import os
import pdb
import torch

from load_mnist import load_mnist, plot_flat_digit


def lin(X, W, b):
    return X@W + b

def relu(x):
    # Try to use single pytorch calls. Fast code implemented in C!
    return x.clamp_min(0.)

def mse(inp, target):
    return (inp - target.squeeze(-1)).pow(2).mean()

def mse_grad(inp, target):
    inp.grad = 2*(inp.squeeze(-1) - target).unsqueeze(-1) / inp.shape[0]

def relu_grad(inp, out):
    inp.grad = (inp>0).float() * out.grad

def lin_grad(inp, out, w, b):
    inp.grad = out @ w.t()
    w.grad = (inp.unsqueeze(-1) * out.grad.unsqueeze(1)).sum(0)
    b.grad = out.grad.sum(0)

def forward_and_backward(inp, target, w1, b1, w2, b2):
    # forward pass
    l1 = lin(inp, w1, b1)
    l2 = relu(l1)
    out = lin(l2, w2, b2)
    loss = mse(out, target)

    # backward pass
    mse_grad(out, target)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)

    print("Loss Function", loss)

def basic_arquitecture(X_train, y_train, X_valid, y_valid):
    """ Implementation of a basic arquitecture.
    
    Input Layer > Hidden Layer > Output Layer

    c: number of activations. Normally we would want to use cross-entropy against the 10 activations but to simplify for now
          we are going to use MSE that means we are gonna have one activation.
    n: Num of examples (50.000)
    m: num of columns (784 pixels)
    """

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

    forward_and_backward(X_train, y_train, w1, b1, w2, b2)

def main():
    # -----------------------------------------
    # Prepare Folders and load mnist
    # -----------------------------------------

    inputs_folder = 'inputs/'
    outputs_folder = 'outputs/'

    for f in [inputs_folder, outputs_folder]:
        os.makedirs(f, exist_ok=True)

    # Load mnist into tensors
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_mnist(inputs_folder)

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

    #plot_digit(train_images[0], train_labels[0])
if __name__ == "__main__":
    main()