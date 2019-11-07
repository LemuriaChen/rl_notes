
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt

# np.random.seed(100)

ACTIONS_REPR = ['up', 'down', 'left', 'right']
ACTIONS = [0, 1, 2, 3]

MAZE_LENGTH = 9
MAZE_WIDTH = 6

INIT_STATE = (3, 0)
TERMINATED_STATE = (8, 5)


def block_maze_step(current_state, current_action, obstacle_states):
    """
    :param current_state: current state
    :param current_action: current action
    :param obstacle_states: states of blocking maze
    :return: next state and reward
    """
    x, y = current_state
    if current_action == 0:     # up
        y = y + 1
    elif current_action == 1:   # down
        y = y - 1
    elif current_action == 2:   # left
        x = x - 1
    elif current_action == 3:   # right
        x = x + 1
    # consider marginal situations
    x = min(max(x, 0), MAZE_LENGTH - 1)
    y = min(max(y, 0), MAZE_WIDTH - 1)
    next_state = (x, y)
    if next_state in obstacle_states:
        return current_state, 0
    elif next_state == TERMINATED_STATE:
        return next_state, 1
    else:
        return next_state, 0


def greedy_policy(q_value_func, current_state, available_actions, eps):
    """
    :param q_value_func: current state action value function
    :param current_state: current state
    :param available_actions: available actions list of current state
    :param eps: epsilon of greedy policy, set to 0 if no greedy policy
    :return: a action chosen by epsilon policy of current state action value function
    """
    prob = [eps / len(available_actions)] * len(available_actions)
    action_space = q_value_func[current_state][available_actions]
    # if more than one action has the same value, then randomly select one action
    arg_max_index = np.where(action_space == max(action_space))[0]
    prob[np.random.choice(arg_max_index)] = 1 - eps + eps / len(available_actions)
    return np.random.choice(available_actions, p=prob)


def show_optimal_path(old_state, q_value_func, obstacle_states):
    step = 0
    print('Game start')
    while True:
        old_action = greedy_policy(q_value_func, old_state, ACTIONS, eps=1e-2)
        print(f'{old_state} -> {ACTIONS_REPR[old_action]}')
        new_state, reward = block_maze_step(old_state, old_action, obstacle_states)
        step += 1
        if new_state == TERMINATED_STATE:
            print(f'{new_state}')
            break
        old_state = new_state
    print(f'Game over, {step} steps used.')


"""
# model learning (actually state action pairs storage)
model = defaultdict(tuple)
model_sa = defaultdict(set)
model_sa_cnt = defaultdict(int)

old_obs_states = [(_, 2) for _ in range(0, 8)]
new_obs_states = [(_, 2) for _ in range(1, 9)]

q_value_func = np.zeros([MAZE_LENGTH, MAZE_WIDTH, 4])

eps = 0.1
gamma = 0.95
alpha = 0.9
n = 10

old_state = INIT_STATE
cum_reward = 0
cum_rewards = [cum_reward]
cum_steps = [0]

for step in range(3000):

    old_action = greedy_policy(q_value_func, old_state, ACTIONS, eps)
    new_state, reward = None, None
    if step <= 1000:
        new_state, reward = block_maze_step(old_state, old_action, old_obs_states)
    else:
        new_state, reward = block_maze_step(old_state, old_action, new_obs_states)

    cum_reward += reward
    cum_rewards.append(cum_reward)
    cum_steps.append(step)

    # update state action value function (direct reinforcement learning)
    q_value_func[old_state][old_action] += alpha * (
            reward + gamma * np.max(q_value_func[new_state]) -
            q_value_func[old_state][old_action]
    )

    model_sa[old_state].add(old_action)
    model[(old_state, old_action)] = (new_state, reward)

    rand_state = None
    # planning (indirect reinforcement learning)
    for loop in range(n):
        rand_state = random.sample(model_sa.keys(), k=1)[0]
        rand_action = random.sample(model_sa[rand_state], k=1)[0]
        new_rand_state, new_reward = model[(rand_state, rand_action)]
        new_rand_state, new_reward = deepcopy(new_rand_state), deepcopy(new_reward)
        q_value_func[rand_state][rand_action] += alpha * (
                new_reward + gamma * np.max(q_value_func[new_rand_state]) -
                q_value_func[rand_state][rand_action]
        )

    if new_state == TERMINATED_STATE:
        old_state = INIT_STATE
    else:
        old_state = new_state


plt.plot(cum_steps, cum_rewards)
plt.show()
"""


# Dyna-Q
def episode(q_value_func, obstacle_states, eps):
    # global model_sa_cnt
    step = 0
    old_state = INIT_STATE
    while True:                                                                             # (a)
        old_action = greedy_policy(q_value_func, old_state, ACTIONS, eps)                   # (b)
        new_state, reward = block_maze_step(old_state, old_action, obstacle_states)         # (c)
        # update state action value function (direct reinforcement learning)
        q_value_func[old_state][old_action] += alpha * (                                    # (d)
                reward + gamma * np.max(q_value_func[new_state]) -
                q_value_func[old_state][old_action]
        )
        model_sa[old_state].add(old_action)                                                 # (e)
        model[(old_state, old_action)] = (new_state, reward)
        model_sa_cnt.__iadd__(1)
        model_sa_cnt[old_state][old_action] = 0

        # planning (indirect reinforcement learning)
        for loop in range(n):                                                               # (f)
            rand_state = random.sample(model_sa.keys(), k=1)[0]
            rand_action = random.sample(model_sa[rand_state], k=1)[0]
            new_rand_state, new_reward = model[(rand_state, rand_action)]
            new_reward += kappa * np.sqrt(model_sa_cnt[rand_state][rand_action])
            q_value_func[rand_state][rand_action] += alpha * (
                    new_reward + gamma * np.max(q_value_func[new_rand_state]) -
                    q_value_func[rand_state][rand_action]
            )
        step += 1
        if new_state == TERMINATED_STATE:
            break
        old_state = new_state
    return step


if __name__ == '__main__':

    # model learning (actually state action pairs storage)
    model = defaultdict(tuple)
    model_sa = defaultdict(set)
    model_sa_cnt = np.zeros([MAZE_LENGTH, MAZE_WIDTH, 4])

    old_obs_states = [(_, 2) for _ in range(0, 8)]
    new_obs_states = [(_, 2) for _ in range(1, 9)]

    q_values = np.zeros([MAZE_LENGTH, MAZE_WIDTH, 4])

    epsilon = 0.1
    gamma = 0.95
    alpha = 1
    kappa = 1e-4
    n = 10

    cum_reward = 0
    cum_rewards = [cum_reward]
    cum_step = 0
    cum_steps = [cum_step]

    while True:
        if cum_step <= 1000:
            cum_step += episode(q_values, old_obs_states, eps=epsilon)
        else:
            # show_optimal_path(INIT_STATE, q_values, old_obs_states)
            cum_step += episode(q_values, new_obs_states, eps=epsilon)

        cum_steps.append(cum_step)
        cum_reward += 1
        cum_rewards.append(cum_reward)

        if cum_step >= 3000:
            break

    plt.plot(cum_steps, cum_rewards)
    plt.xlim(0, 3000)
    plt.show()

    # show_optimal_path(INIT_STATE, q_values, new_obs_states)









