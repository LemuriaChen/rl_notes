

import numpy as np
import matplotlib.pyplot as plt


epsilon_ = 0.1
alpha = 0.5
gamma = 1
wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

action_left = 0
action_right = 1
action_up = 2
action_down = 3
actions = [action_left, action_right, action_up, action_down]
actions_str = ['L', 'R', 'U', 'D']

width = 7
length = 10
state_action_values = np.zeros([length, width, len(actions)])

terminal_state = (7, 3)


def windy_step(previous_state, action):
    x, y = previous_state
    if action == action_left:
        y = y + wind_strength[x]
        x = x - 1
    elif action == action_right:
        y = y + wind_strength[x]
        x = x + 1
    elif action == action_up:
        y = y + wind_strength[x] + 1
    elif action == action_down:
        y = y + wind_strength[x] - 1
    else:
        assert action in actions
    x = min(max(x, 0), 9)
    y = min(max(y, 0), 6)
    return x, y


def greedy_policy(current_state, epsilon):
    prob = [epsilon / len(actions)] * 4
    arg_min_index = np.where(state_action_values[current_state] == max(state_action_values[current_state]))[0]
    prob[np.random.choice(arg_min_index)] = 1 - epsilon + epsilon / len(actions)
    return prob


def episode(verbose=False, epsilon=epsilon_):

    # SARSA (on-policy TD control)
    step = 0

    old_state = (0, 3)
    # take action and observe
    act = np.random.choice(actions, p=greedy_policy(old_state, epsilon))

    while True:
        if verbose:
            print(f'{old_state} -> {act}')
        new_state = windy_step(old_state, act)
        next_act = np.random.choice(actions, p=greedy_policy(new_state, epsilon))
        state_action_values[old_state][act] += alpha * (
                -1 + gamma * state_action_values[new_state][next_act] -
                state_action_values[old_state][act])
        old_state = new_state
        act = next_act
        if new_state == terminal_state:
            break
        step += 1

    if verbose:
        print(f'{old_state}\nend')

    return step


episode_steps = []

for _ in range(500):
    step = episode()
    episode_steps.append(step)

plt.plot(np.cumsum(episode_steps), list(range(len(episode_steps))))
plt.show()


episode(verbose=True, epsilon=0)

# best policy
for i in range(10):
    for j in range(7):
        print(actions_str[np.random.choice(actions, p=greedy_policy((i, j), 0))], end='\t')
    print('')

