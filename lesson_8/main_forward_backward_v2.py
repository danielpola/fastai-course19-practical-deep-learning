import numpy as np
import math
import os
import pdb
import time
import torch

from load_mnist import load_mnist, plot_flat_digit

# Copy of Jeremy's refactoring
class Module():
    # With __call__ we can call a class as a function.
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self): raise Exception('not implemented')
    def backward(self): self.bwd(self.out, *self.args)

class Relu(Module):
    def forward(self, inp): return inp.clamp_min(0.)-0.5
    def bwd(self, inp, out): inp.grad = (inp.grad > 0).float() * out.grad

class Lin(Module):
    def __init__(self, w, b): self.w, self.b = w,b
    def forward(self, inp): return inp@self.w + self.b
    def bwd(self, inp, out): 
        inp.grad = out @ self.w.t()
        self.w.grad = torch.einsum("bi,bj->ij", inp, out.grad)
        self.b.grad = out.grad.sum(0)

class Mse(Module):
    def forward(self, inp, target): return (inp - target.squeeze(-1)).pow(2).mean()
    def bwd(self, inp, out, target): inp.grad = 2*(inp.squeeze(-1) - target).unsqueeze(-1) / inp.shape[0]

class Model():
    def __init__(self, w1, b1, w2, b2):
        self.layers = [Lin(w1,b1), Relu(), Lin(w2,b2)]
        self.loss = Mse()

    def __call__(self, x, target):
        for l in self.layers: x = l(x)
        return self.loss(x, target)
    
    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers): l.backward()


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

    w1.grad, b1.grad, w2.grad, b2.grad = [None] * 4

    t0 = time.time()
    model = Model(w1,b1,w2,b2)
    model(X_train, y_train)

    print(time.time() - t0)

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