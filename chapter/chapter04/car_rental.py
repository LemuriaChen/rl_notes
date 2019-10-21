
from scipy.stats import poisson
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# part of variables naming rule reference:
# https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter04/car_rental.py

# Example 4.2: Jack’s Car Rental
# assume the first place is A and the second place is B
MAX_CARS = 20
MAX_MOVE_OF_CARS = 5
RENTAL_CREDIT = 10
MOVE_CAR_COST = 2

RENTAL_A = 3
RENTAL_B = 4
RETURNS_A = 3
RETURNS_B = 2

DISCOUNT = 0.9
POISSON_UPPER_BOUND = 11

# state set (# of cars in the two places, [0, 20] * [0, 20])
STATE_SET = [(a, b) for a in range(0, MAX_CARS + 1) for b in range(0, MAX_CARS + 1)]
# action set (1 means moving 1 car from A to B, -1 means moving 1 car from B to A)
ACTION_SET = list(range(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS, 1))

# initial policy
init_policy = {key: 0 for key in STATE_SET}
# initial state value function
init_value = {key: 0 for key in STATE_SET}
# cached probability mass function
PMF = {_: {k: poisson.pmf(k, _) for k in range(MAX_CARS + 1)}
       for _ in set(list([RENTAL_A, RENTAL_B, RETURNS_A, RETURNS_B]))}


# policy iteration (two modules)
def state_value_func(state: tuple, value: dict, action: int) -> int:
    """
    :param state: a tuple of state
    :param value: a dict of state value function which map state to expected cumulative reward
    :param action: a number of which action to take
    :return: expected rewards
    """
    num_cars_a = max(min(state[0] - action, MAX_CARS), 0)
    num_cars_b = max(min(state[1] + action, MAX_CARS), 0)
    expected_returns = - abs(action) * MOVE_CAR_COST
    # theoretically, POISSON_UPPER_BOUND should be infinite
    # a upper limit is used here to reduce meaningless operation
    for rental_a in range(POISSON_UPPER_BOUND):
        for rental_b in range(POISSON_UPPER_BOUND):
            for return_a in range(POISSON_UPPER_BOUND):
                for return_b in range(POISSON_UPPER_BOUND):
                    # probability of occurrence of independent events
                    prob = PMF.get(RENTAL_A).get(rental_a) * PMF.get(RENTAL_B).get(rental_b) * \
                           PMF.get(RETURNS_A).get(return_a) * PMF.get(RETURNS_B).get(return_b)
                    # reward of rental cars
                    reward = (min(rental_a, num_cars_a) + min(rental_b, num_cars_b)) * RENTAL_CREDIT
                    # based on the bellman equation, see this in p102, black box.
                    # some of 'prob' here is little smaller than the actual value
                    expected_returns += prob * (reward + DISCOUNT * value[
                        min(num_cars_a - min(rental_a, num_cars_a) + return_a, MAX_CARS),
                        min(num_cars_b - min(rental_b, num_cars_b) + return_b, MAX_CARS)
                    ])
    return expected_returns


# policy evaluation: given policy, evaluate state/action value function
def policy_eval(policy: dict, value: dict) -> dict:
    """
    :param policy: a dict of policy which map state to action
    :param value: a dict of state value function which map state to expected cumulative reward
    :return: a updated dict of state value function given a policy
    """
    # assume current state is (10, 10)
    # and the actions is +1, then the state is (9, 11)
    # rental of A can be 0 to +oo, assume it's 1
    # rental of B can be 0 to +oo, assume it's 2
    # return of A can be 0 to +oo, assume it's 3
    # return of B can be 0 to +oo, assume it's 2
    # thus the succeed state is (9 - 1 + 3, 11 - 2 + 2), i.e., (11, 11)
    # then the expected reward is:
    # p(1, RENTAL_A) * p(2, RENTAL_B) * p(3, RETURNS_A) * p(2, RETURNS_B) *
    # ((1 + 2) * RENTAL_CREDIT - 1 * MOVE_CAR_COST + state(11, 11))
    num_iterations = 0
    while True:
        # store a backup of state value function
        old_value = value.copy()
        for state in STATE_SET:
            # in-place state value function iteration
            action = policy.get(state)
            value[state] = state_value_func(state, value, action)
        num_iterations += 1
        max_value_change = max([abs(value.get(key) - old_value.get(key)) for key in value])
        print(f'max value change {max_value_change}')
        if max_value_change < 1e-4:
            print(f'state value function converges in {num_iterations} iterations')
            break
    return value


# policy improvement: given state/action value function, find greedy policy
def policy_improved(value: dict) -> dict:
    """
    :param value: a dict of state value function which map state to expected cumulative reward
    :return: a greedy policy based on the state value function
    """
    policy = {}
    for state in tqdm(STATE_SET):
        state_value_list = []
        for action in ACTION_SET:
            if (0 <= action <= state[0]) or (-state[1] <= action <= 0):
                state_value_list.append(
                    state_value_func(state, value, action))
            else:
                state_value_list.append(-np.Inf)
        policy[state] = ACTION_SET[np.array(state_value_list).argmax()]
    return policy


def visualize(data_matrix):
    figure = plt.figure()
    figure.set_tight_layout(True)
    ax = Axes3D(figure)
    x, y = np.meshgrid(np.arange(0, MAX_CARS + 1), np.arange(0, MAX_CARS + 1))
    ax.plot_surface(x, y, data_matrix, cmap='rainbow')
    plt.show()


iterations = 0

while True:
    iterations += 1

    print(f'\nthe {iterations} iteration of policy evaluation')
    new_value = policy_eval(init_policy, init_value)
    print(f'\nthe {iterations} iteration of policy improvement')
    new_policy = policy_improved(new_value)

    if new_policy == init_policy:
        break

    init_policy = new_policy
    init_value = new_value

value_matrix = np.zeros([MAX_CARS + 1, MAX_CARS + 1])
policy_matrix = np.zeros([MAX_CARS + 1, MAX_CARS + 1])

for i in range(MAX_CARS + 1):
    for j in range(MAX_CARS + 1):
        value_matrix[i][j] = new_value.get((i, j))
        policy_matrix[i][j] = new_policy.get((i, j))

# visualization of state value function
visualize(value_matrix)

# visualization of policy
visualize(policy_matrix)
