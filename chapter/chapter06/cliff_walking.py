
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


action_left = 0
action_right = 1
action_up = 2
action_down = 3

actions_set = [action_left, action_right, action_up, action_down]
actions_rep = ['left', 'right', 'up', 'down']
actions_set_len = len(actions_set)


gamma = 1
epsilon = 0.1
alpha = 0.5

length = 12
width = 4

cliff_state_sets = [(_, 0) for _ in range(1, 11)]
init_state = (0, 0)
terminate_state = (11, 0)


def cliff_step(current_state: tuple, action: int) -> tuple:
    """
    print(f'(0,0) -> left  -> {cliff_step((0, 0), 0)}')
    print(f'(0,0) -> right -> {cliff_step((0, 0), 1)}')
    print(f'(0,0) -> up -> {cliff_step((0, 0), 2)}')
    print(f'(0,0) -> down -> {cliff_step((0, 0), 3)}')
    """
    x, y = current_state
    if action == action_left:
        x = x - 1
    elif action == action_right:
        x = x + 1
    elif action == action_up:
        y = y + 1
    elif action == action_down:
        y = y - 1
    else:
        assert action in actions_set
    x = min(max(x, 0), length - 1)
    y = min(max(y, 0), width - 1)
    return x, y


def greedy_policy(current_state: tuple, eps: float):
    """
        if eps = 0, then the greedy policy is a deterministic policy
    """
    prob = [eps / actions_set_len] * actions_set_len
    arg_min_index = np.where(state_action_values[current_state] == max(
        state_action_values[current_state]))[0]
    prob[np.random.choice(arg_min_index)] = 1 - eps + eps / actions_set_len
    return prob


def episode_q_learning(update: bool, eps: float, verbose: bool):
    """
    :param update:
    :param eps:
    :param verbose:
    :return:
    """
    old_state = init_state
    total_rewards = 0

    while True:
        old_action = np.random.choice(actions_set,
                                      p=greedy_policy(old_state, eps))
        if verbose:
            print(f'{old_state} -> {old_action}')
        new_state = cliff_step(old_state, old_action)

        reward = -100 if new_state in cliff_state_sets else -1
        total_rewards += reward

        new_state = init_state if new_state in cliff_state_sets else new_state

        if update:
            state_action_values[old_state][old_action] += alpha * (
                reward + gamma * np.max(state_action_values[new_state]) - state_action_values[old_state][old_action]
            )

        if new_state == terminate_state:
            break

        old_state = new_state

    return total_rewards


def episode_sarsa(update: bool, eps: float, verbose: bool):

    old_state = init_state
    old_action = np.random.choice(actions_set,
                                  p=greedy_policy(old_state, eps))
    total_rewards = 0

    while True:
        if verbose:
            print(f'{old_state} -> {old_action}')

        new_state = cliff_step(old_state, old_action)

        reward = -100 if new_state in cliff_state_sets else -1
        total_rewards += reward

        new_state = init_state if new_state in cliff_state_sets else new_state
        new_action = np.random.choice(actions_set, p=greedy_policy(new_state, eps))

        if update:
            state_action_values[old_state][old_action] += alpha * (
                reward + gamma * state_action_values[new_state][new_action] -
                state_action_values[old_state][old_action]
            )

        if new_state == terminate_state:
            break

        old_state = new_state
        old_action = new_action

    return total_rewards


################################################################################
# Q-learning
n_episodes = 500
n_avg = 40
rewards_q_learning = np.zeros([n_avg, n_episodes])

for i in tqdm(range(n_avg)):
    state_action_values = np.zeros([length, width, len(actions_set)])
    for j in range(n_episodes):
        rewards_q_learning[i, j] = episode_q_learning(update=True, eps=epsilon, verbose=False)

episode_q_learning(update=False, eps=0, verbose=True)

################################################################################
# SARSA
n_episodes = 500
n_avg = 40
rewards_sarsa = np.zeros([n_avg, n_episodes])

for i in tqdm(range(n_avg)):
    state_action_values = np.zeros([length, width, len(actions_set)])
    for j in range(n_episodes):
        rewards_sarsa[i, j] = episode_sarsa(update=True, eps=epsilon, verbose=False)

episode_sarsa(update=False, eps=0, verbose=True)

################################################################################
plt.plot(rewards_q_learning.mean(axis=0), label='Q-learning')
plt.plot(rewards_sarsa.mean(axis=0), label='SARSA')
plt.ylim(-100, 0)
plt.legend()
plt.show()



