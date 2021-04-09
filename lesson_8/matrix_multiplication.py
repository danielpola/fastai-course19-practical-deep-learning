#%%
# Matrix multiplication with python and pytorch

# Run this with iPython to use %timeit. I use the vscode integrated notebooks.

import numpy as np
import torch
from torch import tensor


def near(a,b):
    return torch.allclose(a, b, rtol=1e-3, atol=1e-5)

def py_naive_mat_mul(a, b):
    """ Python naive implementation of 2D matrix multiplication.
    a and b are tensors of rank 2 ."""

    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape

    assert a_cols == b_rows, f"cannot multiply matrices of shapes {a.shape} and {b.shape}"

    c = torch.zeros((a_rows, b_cols))
    # Loop rows of a
    for i in range(a_rows):
        # Loop columns of b
        for j in range(b_cols):
            # inline loop of a_row and b_col multiplication
            c[i][j] = sum([x*y for (x,y) in zip(a[i,:], b[:,j])])

    return c


def broadcasted_mat_mul(a, b):
    """ Python naive implementation of 2D matrix multiplication.
    a and b are tensors of rank 2 ."""

    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape

    assert a_cols == b_rows, f"cannot multiply matrices of shapes {a.shape} and {b.shape}"

    c = torch.zeros((a_rows, b_cols))
    # Loop rows of a
    for i in range(a_rows):
        # Replacing just one loop we have
        # # Loop columns of b
        # for j in range(b_cols):
        #     # inline loop of a_row and b_col multiplication
        #     c[i][j] = (a[i,:] * b[:,j]).sum()

        # Traspose a row and broadcast to multiply b columns
        # a[i,:].unsqueeze(-1) Gives a vertical representation of the i-th row
        # * is a broadcasted element wise multiplication
        c[i,:] = (a[i,:].unsqueeze(-1) * b).sum(axis=0)

    return c

#%%
# Create two matrices with arbitrary dimensions
a = torch.rand(10,7)
b = torch.rand(7,3)

# Expected result from numpy library
c_reference = torch.matmul(a, b)
print("Timeit torch reference:")
%timeit -n 100 torch.matmul(a, b)

time_reference = 4.22 # micro seconds => 4.22 µs ± 1.19 µs

#%%

# Naive implementation: Result comparison and time
c_naive = py_naive_mat_mul(a, b)
assert near(c_reference, c_naive), "Matmul implementation returns a different result than expected"
print("Timeit naive:")
%timeit -n 100 py_naive_mat_mul(a, b)

time_naive = 1.89 * 1000 # micro seconds => 1.89 ms ± 43 µs per loop

#%%
# Broadcasted implementation: Result comparison and time
c_broad = broadcasted_mat_mul(a, b)
assert near(c_reference, c_broad), "Matmul implementation returns a different result than expected"
print("Timeit broad:")
%timeit -n 100 broadcasted_mat_mul(a, b)

time_broad = 178# 178 µs ± 22.7 µs per loop

# We are allowed to use torch matmul from now on
