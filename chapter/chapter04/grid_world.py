
import numpy as np


# iterative policy evaluation

# initial random guess
x_k = np.zeros(16)
epsilon = 1e-10

# 0   1   2   3
# 4   5   6   7
# 8   9   10  11
# 12  13  14  15

k = 0
while True:
    x_k_1 = np.zeros(16)
    x_k_1[1] = (x_k[0] - 1 + x_k[2] - 1 + x_k[5] - 1 + x_k[1] - 1) / 4
    x_k_1[2] = (x_k[1] - 1 + x_k[3] - 1 + x_k[6] - 1 + x_k[2] - 1) / 4
    x_k_1[3] = (x_k[2] - 1 + x_k[7] - 1 + x_k[3] - 1 + x_k[3] - 1) / 4
    x_k_1[4] = (x_k[0] - 1 + x_k[5] - 1 + x_k[8] - 1 + x_k[4] - 1) / 4
    x_k_1[5] = (x_k[1] - 1 + x_k[4] - 1 + x_k[6] - 1 + x_k[9] - 1) / 4
    x_k_1[6] = (x_k[2] - 1 + x_k[5] - 1 + x_k[7] - 1 + x_k[10] - 1) / 4
    x_k_1[7] = (x_k[3] - 1 + x_k[6] - 1 + x_k[11] - 1 + x_k[7] - 1) / 4
    x_k_1[8] = (x_k[4] - 1 + x_k[8] - 1 + x_k[9] - 1 + x_k[12] - 1) / 4
    x_k_1[9] = (x_k[5] - 1 + x_k[8] - 1 + x_k[10] - 1 + x_k[13] - 1) / 4
    x_k_1[10] = (x_k[6] - 1 + x_k[9] - 1 + x_k[11] - 1 + x_k[14] - 1) / 4
    x_k_1[11] = (x_k[7] - 1 + x_k[10] - 1 + x_k[15] - 1 + x_k[11] - 1) / 4
    x_k_1[12] = (x_k[8] - 1 + x_k[13] - 1 + x_k[12] - 1 + x_k[12] - 1) / 4
    x_k_1[13] = (x_k[9] - 1 + x_k[12] - 1 + x_k[14] - 1 + x_k[13] - 1) / 4
    x_k_1[14] = (x_k[10] - 1 + x_k[13] - 1 + x_k[15] - 1 + x_k[14] - 1) / 4
    if np.linalg.norm(x_k - x_k_1) < epsilon:
        break
    if k in [0, 1, 2, 3, 10]:
        print(f'\nk = {k}')
        print(x_k.reshape(4, 4).round(2))
    k = k + 1
    x_k = x_k_1

# page 99, Figure 4.1
print(f'\nk = oo')
print(x_k.reshape(4, 4))
