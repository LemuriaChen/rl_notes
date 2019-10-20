
# Grid world example

import numpy as np
from scipy.linalg import solve
from scipy.optimize import fsolve


def bound_value(x, min_val=0, max_val=4):
    return min(max(x, min_val), max_val)


cof_matrix = []     # coefficient matrix of Bellman equation (25 * 25)
const = []          # constant term of Bellman equation (25 * 1)
grid_size = 5
fix_const = .9 / 4

# solve Bellman equation for state value function (by solving linear equations)
# there is also some other algorithms, but in order to understand Bellman equation, I use linear equations here.

# bellman equation
for i in range(grid_size):
    for j in range(grid_size):
        cof = np.zeros([grid_size, grid_size])
        cof[i, j] += -1
        if i == 0 and j == 1:
            cof[4, 1] = 4 * fix_const
            const.append(- 40 / 4)
        elif i == 0 and j == 3:
            cof[2, 3] = 4 * fix_const
            const.append(- 20 / 4)
        else:
            cof[bound_value(i - 1), bound_value(j)] += fix_const
            cof[bound_value(i + 1), bound_value(j)] += fix_const
            cof[bound_value(i), bound_value(j - 1)] += fix_const
            cof[bound_value(i), bound_value(j + 1)] += fix_const
            const.append(
                ((i == 0 or i == grid_size - 1) + (j == 0 or j == grid_size - 1)) / 4
            )
        cof_matrix.append(list(cof.reshape(-1)))

sol = solve(cof_matrix, const)

# page 82, Figure 3.2
# print(np.array(cof_matrix))
# print(np.array(const))
print('state value function of GridWorld under a random policy:')
print(np.array(sol).reshape([grid_size, grid_size]).round(1))


# bellman optimality equation
def bellman_optimality_function(x, gamma=0.9):

    # 0   1   2   3   4
    # 5   6   7   8   9
    # 10  11  12  13  14
    # 15  16  17  18  19
    # 20  21  22  23  24

    # this function can be more automated instead of writing equations manually
    # for easier understanding, we remain unchanged for the time being

    return np.array([
        x[0] - max(x[1] * gamma, x[5] * gamma, x[0] * gamma - 1, ),
        x[1] - (x[21] * gamma + 10),
        x[2] - max(x[1] * gamma, x[3] * gamma, x[7] * gamma, x[2] * gamma - 1, ),
        x[3] - (x[13] * gamma + 5),
        x[4] - max(x[3] * gamma, x[9] * gamma, x[4] * gamma - 1, ),
        x[5] - max(x[0] * gamma, x[6] * gamma, x[10] * gamma, x[5] * gamma - 1, ),
        x[6] - max(x[1] * gamma, x[5] * gamma, x[7] * gamma, x[11] * gamma, ),
        x[7] - max(x[2] * gamma, x[6] * gamma, x[8] * gamma, x[12] * gamma, ),
        x[8] - max(x[3] * gamma, x[7] * gamma, x[9] * gamma, x[13] * gamma, ),
        x[9] - max(x[4] * gamma, x[8] * gamma, x[14] * gamma, x[9] * gamma - 1, ),
        x[10] - max(x[5] * gamma, x[11] * gamma, x[15] * gamma, x[10] * gamma - 1, ),
        x[11] - max(x[6] * gamma, x[10] * gamma, x[16] * gamma, x[12] * gamma, ),
        x[12] - max(x[7] * gamma, x[11] * gamma, x[17] * gamma, x[13] * gamma, ),
        x[13] - max(x[8] * gamma, x[12] * gamma, x[18] * gamma, x[14] * gamma, ),
        x[14] - max(x[9] * gamma, x[13] * gamma, x[19] * gamma, x[14] * gamma - 1, ),
        x[15] - max(x[10] * gamma, x[20] * gamma, x[16] * gamma, x[15] * gamma - 1, ),
        x[16] - max(x[11] * gamma, x[15] * gamma, x[21] * gamma, x[17] * gamma, ),
        x[17] - max(x[12] * gamma, x[16] * gamma, x[22] * gamma, x[18] * gamma, ),
        x[18] - max(x[13] * gamma, x[17] * gamma, x[23] * gamma, x[19] * gamma, ),
        x[19] - max(x[14] * gamma, x[18] * gamma, x[24] * gamma, x[19] * gamma - 1, ),
        x[20] - max(x[15] * gamma, x[21] * gamma, x[20] * gamma - 1, ),
        x[21] - max(x[16] * gamma, x[20] * gamma, x[22] * gamma, x[21] * gamma - 1),
        x[22] - max(x[17] * gamma, x[21] * gamma, x[23] * gamma, x[22] * gamma - 1),
        x[23] - max(x[18] * gamma, x[22] * gamma, x[24] * gamma, x[23] * gamma - 1),
        x[24] - max(x[19] * gamma, x[23] * gamma, x[24] * gamma - 1),
    ])


"""
x = np.array([
    22.0, 24.4, 22.0, 19.4, 17.5,
    19.8, 22.0, 19.8, 17.8, 16.0,
    17.8, 19.8, 17.8, 16.0, 14.4,
    16.0, 17.8, 16.0, 14.4, 13.0,
    14.4, 16.0, 14.4, 13.0, 11.7
])
print(bellman_optimality_function(x).reshape([grid_size, grid_size]).round(1))
"""

# solve non-linear equation
sol = fsolve(bellman_optimality_function, x0=np.array([1]*grid_size*grid_size))

# page 87, Figure 3.5
print('\noptimal state value function of GridWorld:')
print(np.array(sol).reshape([grid_size, grid_size]).round(1))


# iteration method (jacobi iteration)
cof_matrix = np.array(cof_matrix)
const = np.array(const)
D = np.diag(cof_matrix)
L = np.tril(cof_matrix, k=-1)
U = np.triu(cof_matrix, k=1)
inv_D = np.diag(1/D)
assert np.array_equal(np.diag(D) + L + U, cof_matrix)

# initial random guess
x_k = np.random.rand(25)
epsilon = 1e-3

while True:
    # update solution by jacobi iteration, of course there are other more complex iterative methods
    # the convergence is guaranteed by Banach's fixed point theorem
    x_k_1 = inv_D.dot(const - (L+U).dot(x_k))
    if np.linalg.norm(x_k_1 - x_k) < epsilon:
        break
    x_k = x_k_1
x_final = x_k_1
print('\niterative policy evaluation by jacobi iteration:')
print(x_final.reshape([grid_size, grid_size]).round(1))

