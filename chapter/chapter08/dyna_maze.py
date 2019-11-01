
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


obstacle_states = [
    (2, 2), (2, 3), (2, 4), (5, 1), (7, 3), (7, 4), (7, 5)
]
init_state = (0, 3)
terminal_state = (8, 5)
actions = [0, 1, 2, 3]
alpha = 0.1
gamma = 0.95
epsilon = 0.1
actions_str = ['left', 'right', 'up', 'down']


def maze_step(current_state: tuple, current_action: int):
    """
    :param current_state: a tuple of current state
    :param current_action: a numeric value of current action
    :return: a tuple of next state and reward
        print(maze_step((1, 2), 0))
        print(maze_step((1, 2), 1))
        print(maze_step((1, 2), 2))
        print(maze_step((1, 2), 3))
    """
    x, y = current_state
    if current_action == 0:  # left
        x = x - 1
    elif current_action == 1:  # right
        x = x + 1
    elif current_action == 2:  # up
        y = y + 1
    elif current_action == 3:  # down
        y = y - 1
    x = min(max(x, 0), 8)
    y = min(max(y, 0), 5)
    next_state = (x, y)
    if next_state in obstacle_states:
        return current_state, 0
    elif next_state == terminal_state:
        return next_state, 1
    else:
        return next_state, 0


def greedy_policy(current_state, eps):
    prob = [eps / len(actions)] * len(actions)
    arg_min_index = np.where(state_values[current_state] == max(state_values[current_state]))[0]
    prob[np.random.choice(arg_min_index)] = 1 - eps + eps / len(actions)
    return prob


def episode(eps, update=True, verbose=False):
    old_state = init_state
    step = 0
    while True:
        old_action = np.random.choice(actions, p=greedy_policy(old_state, eps))
        if verbose:
            print(f'{old_state} -> {actions_str[old_action]}')
        new_state, reward = maze_step(old_state, old_action)
        if update:
            state_values[old_state][old_action] += alpha * (
                reward + gamma * np.max(state_values[new_state]) - state_values[old_state][old_action]
            )
        step += 1
        if new_state == terminal_state:
            break
        old_state = new_state
    return step


# steps = np.zeros([10, 50])
# for i in tqdm(range(10)):
#     state_values = np.zeros([9, 6, 4])
#     for j in range(50):
#         steps[i, j] = episode(epsilon)
#
# plt.plot(steps.mean(axis=0))
# plt.show()


state_values = np.zeros([9, 6, 4])
for _ in range(100):
    episode(epsilon)

episode(eps=0, update=False, verbose=True)


all_states = [(i, j) for i in range(9) for j in range(6)]
obstacle_states = [(2, 2), (2, 3), (2, 4), (5, 1), (7, 3), (7, 4), (7, 5)]
available_states = [_ for _ in all_states if _ not in obstacle_states and _ != terminal_state]
state_values = np.zeros([9, 6, 4])
alpha = 0.1
gamma = 0.95


# Random-sample one-step tabular Q-planning
def q_planning():
    old_state = available_states[np.random.randint(low=0, high=len(available_states), size=1)[0]]
    old_action = np.random.choice(actions)
    new_state, reward = maze_step(old_state, old_action)
    state_values[old_state][old_action] += alpha * (
        reward + gamma * np.max(state_values[new_state]) - state_values[old_state][old_action]
    )


for i in range(100000):
    q_planning()


def very_greedy_policy(state_values_func, current_state, eps=0):
    prob = [eps / len(actions)] * len(actions)
    arg_min_index = np.where(state_values_func[current_state] == max(state_values_func[current_state]))[0]
    prob[np.random.choice(arg_min_index)] = 1 - eps + eps / len(actions)
    return prob


def game_start(old_state, state_values_func):
    while True:
        old_action = np.random.choice(actions, p=very_greedy_policy(state_values_func, old_state))
        print(f'{old_state} -> {actions_str[old_action]}')
        new_state, reward = maze_step(old_state, old_action)
        if new_state == terminal_state:
            break
        old_state = new_state


game_start((7, 0), state_values)
