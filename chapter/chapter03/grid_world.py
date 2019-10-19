
# Grid world example

import numpy as np
from scipy.linalg import solve


def bound_value(x, min_val=0, max_val=4):
    return min(max(x, min_val), max_val)


cof_matrix = []     # coefficient matrix of Bellman equation (25 * 25)
const = []          # constant term of Bellman equation (25 * 1)
grid_size = 5
fix_const = .9 / 4

# solve Bellman equation for state value function (by solving linear equations)
# there is also some other algorithms, but in order to understand Bellman equation, I use linear equations here.

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



