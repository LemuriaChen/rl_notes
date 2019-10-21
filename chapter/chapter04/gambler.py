
import numpy as np
import matplotlib.pyplot as plt

state_set = list(range(0, 101))
p_head = 0.4
value_set = {state: 1 if state == 100 else 0 for state in state_set}
step = 0


# value iteration
while True:
    delta = 0
    for state in state_set[1: -1]:
        v = value_set.get(state)
        action_set = list(range(0, min(state, 100 - state) + 1))
        max_expected_returns = -np.Inf
        for action in action_set:
            expected_returns = p_head * value_set.get(state + action) + \
                               (1 - p_head) * value_set.get(state - action)
            if expected_returns > max_expected_returns:
                max_expected_returns = expected_returns
        value_set[state] = max_expected_returns
        delta = max(delta, abs(v - max_expected_returns))
    step += 1
    if step in [1, 2, 3, 32]:
        plt.plot(state_set, [value_set.get(state) for state in state_set], label=f'sweep {step}')
    if delta < 1e-20:
        break

print(f'steps of total iterations: {step}')

plt.plot(state_set, [value_set.get(state) for state in state_set], label=f'final sweep')
plt.legend(loc='best')
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.grid()
plt.show()


# final policy
policy = {}
for state in state_set[1: -1]:
    action_set = list(range(0, min(state, 100 - state) + 1))
    expected_returns_list = []
    for action in action_set:
        expected_returns_list.append(p_head * value_set.get(state + action) +
                                     (1 - p_head) * value_set.get(state - action))
    # there are some problems (wtf...), see
    # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
    policy[state] = action_set[(np.round(expected_returns_list[1:], 5)).argmax() + 1]


# Example 4.3: Gambler’s Problem (Figure 4.3, P 106)
plt.scatter([state for state in policy], [policy.get(state) for state in policy])
plt.xlabel('Capital')
plt.ylabel('Final policy (stake)')
plt.grid()
plt.show()
