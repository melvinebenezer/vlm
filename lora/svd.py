import torch
import numpy as np
_ = torch.manual_seed(0)

d, k  = 10, 10

# THis way we can generate a rank deficient matrix
W_rank = 2
W = torch.randn(d, W_rank) @ torch.randn(W_rank, k)
print(f"W : {W}")

W_rank = np.linalg.matrix_rank(W)
print(f"Rank of W: {W_rank}")

# Calculate the SVD of W matrix

U,S,V = torch.svd(W)

# For rank-r factorization, jeep on the first r singular values(and corresponding columns of u and v)
U_r = u[:, :W_rank]
S_r = torch.diag(S[:W_rank])
V_r = V[:, :W_rank].t()

# Compute B = U_r * S_r and A = V_r
B = U_r @ S_r
A = V_r

print(f"shape of B: {B.shape}")
print(f"shapeof A: {A.shape}")


# Given the same input , check the output using the original W and the matrices resulting from the decomposition

bias = torch.randn(d)
x = torch.randn(d)

#Compute y = Wx + b
y = W @ x + bias

# Compute y_hat = B(Ax) + b
y_hat = ( B @ A) @ x + bias

print("Original y using W: ", y)
print("y_hat using B and A: ", y_hat)

print("Total number of parameters in W: ", W.numel())
print("Total number of parameters in B and A: ", B.numel() + A.numel())