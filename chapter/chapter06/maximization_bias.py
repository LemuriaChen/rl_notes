
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# A_state_action_values = np.array([0, 0])
# B_state_action_values = np.array([0] * len(b_actions))
# A_actions = ['left', 'right']
# B_actions = list(range(10))

states_set = ['A', 'B']
init_state = 'A'
terminate_state = 'T'

actions_sets = {'A': [0, 1],    # 0 means left, 1 means right
                'B': list(range(10))}

epsilon = 0.1
alpha = 0.1
gamma = 1


def simple_step(current_state, action):
    """
        print(simple_step('A', 'left'))
        print(simple_step('A', 'right'))
        print(simple_step('B', 0))
        print(simple_step('B', 1))
    """
    reward = 0
    assert current_state in states_set
    assert action in actions_sets.get(current_state)

    if current_state == 'A':
        if action == 0:
            next_state = 'B'
        else:
            next_state = 'T'
    else:
        next_state = 'T'
        reward = np.random.normal(loc=-0.1, scale=1, size=1)[0]
    return next_state, reward


def greedy_policy(current_state, eps):
    """
        print(greedy_policy('A', epsilon))
        print(greedy_policy('B', epsilon))
    """
    assert current_state in states_set
    total_actions = len(actions_sets.get(current_state))
    prob = [eps / total_actions] * total_actions
    arg_min_index = np.where(state_action_values.get(current_state) ==
                             np.max(state_action_values.get(current_state)))[0]
    prob[np.random.choice(arg_min_index)] = 1 - eps + eps / len(prob)
    return prob


def episode(verbose=False, update=True, eps=epsilon):
    old_state = 'A'
    actions_list = []
    while True:
        old_action = np.random.choice(actions_sets.get(old_state),
                                      p=greedy_policy(old_state, eps=eps))
        actions_list.append(old_action)
        if verbose:
            print(f'{old_state} -> {old_action}')
        new_state, reward = simple_step(old_state, old_action)
        if update:
            state_action_values.get(old_state)[old_action] += alpha * (
                reward + gamma * np.max(state_action_values.get(new_state) -
                                        state_action_values.get(old_state)[old_action])
            )
        if new_state == 'T':
            break
        old_state = new_state
    return actions_list[0]


n_bootstrap = 10000
n_episodes = 300
first_action = np.zeros([n_bootstrap, n_episodes])

for i in tqdm(range(n_bootstrap)):
    state_action_values = {'A': np.zeros(len(actions_sets.get('A'))),
                           'B': np.zeros(len(actions_sets.get('B'))),
                           'T': 0}
    for j in range(n_episodes):
        first_action[i][j] = episode(verbose=False)

# plt.plot(1 - first_action.mean(axis=0), color='b', label='Q-learning')
# plt.axhline(y=0.05, color='r', label='optimal')
# plt.legend()
# plt.show()


# there are some cases that make the optimal strategy not the optimal strategy
# episode(verbose=True, update=False, eps=0)


# Double Q-learning

def greedy_policy_double(current_state, eps):
    assert current_state in states_set
    total_actions = len(actions_sets.get(current_state))
    prob = [eps / total_actions] * total_actions
    q = state_action_values_q1.get(current_state) + state_action_values_q2.get(current_state)
    arg_min_index = np.where(q == np.max(q))[0]
    prob[np.random.choice(arg_min_index)] = 1 - eps + eps / len(prob)
    return prob


def episode_double(verbose=False, update=True, eps=epsilon):
    old_state = 'A'
    actions_list = []
    while True:
        old_action = np.random.choice(actions_sets.get(old_state),
                                      p=greedy_policy_double(old_state, eps=eps))
        actions_list.append(old_action)
        if verbose:
            print(f'{old_state} -> {old_action}')
        new_state, reward = simple_step(old_state, old_action)
        if update:
            if np.random.rand() < 0.5:
                new_action = np.random.choice(
                    np.where(state_action_values_q1.get(new_state) ==
                             np.max(state_action_values_q1.get(new_state)))[0])
                state_action_values_q1.get(old_state)[old_action] += alpha * (
                    reward + gamma * state_action_values_q2.get(new_state)[new_action] -
                    state_action_values_q1.get(old_state)[old_action])
            else:
                new_action = np.random.choice(
                    np.where(state_action_values_q2.get(new_state) ==
                             np.max(state_action_values_q2.get(new_state)))[0])
                state_action_values_q2.get(old_state)[old_action] += alpha * (
                    reward + gamma * state_action_values_q1.get(new_state)[new_action] -
                    state_action_values_q2.get(old_state)[old_action])
        if new_state == 'T':
            break
        old_state = new_state
    return actions_list[0]


n_bootstrap = 10000
n_episodes = 300
first_action_double = np.zeros([n_bootstrap, n_episodes])

for i in tqdm(range(n_bootstrap)):
    state_action_values_q1 = {
        'A': np.zeros(len(actions_sets.get('A'))),
        'B': np.zeros(len(actions_sets.get('B'))),
        'T': [0]}
    state_action_values_q2 = {
        'A': np.zeros(len(actions_sets.get('A'))),
        'B': np.zeros(len(actions_sets.get('B'))),
        'T': [0]}
    for j in range(n_episodes):
        first_action_double[i][j] = episode_double()


plt.plot(1 - first_action.mean(axis=0), color='g', label='Q-learning')
plt.plot(1 - first_action_double.mean(axis=0), color='b', label='Double Q-learning')
plt.axhline(y=0.05, color='r', label='optimal')
plt.legend()
plt.show()


# there are some cases that make the optimal strategy not the optimal strategy
# episode_double(verbose=True, update=False, eps=0)


