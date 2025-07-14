import numpy as np

np.random.seed(42)

A = np.random.rand(2, 4)
B = np.random.rand(2, 4)
B = B.T
C = np.zeros((2, 2))

# C = A * B
for i in range(2):
    for j in range(2):
        for k in range(4):
            C[i, j] += A[i, k] * B[k, j] 

C_np = A @ B

print(C_np)
print(" ")
print(C)