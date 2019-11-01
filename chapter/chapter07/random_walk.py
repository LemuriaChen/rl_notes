
import numpy as np


# Example 7.1: n-step TD Methods on the Random Walk

state_values_real = np.arange(-20, 22, 2) / 20.0

gamma = 1
initial_state = 10
left_terminal_state = 0
right_terminal_state = 20


def mse(state_val):
    return np.sqrt(np.mean(np.square(np.array(state_val[1: -1]) -
                                     np.array(state_values_real[1: -1]))))


def episode(n):

    old_state = initial_state
    step = 0
    states_history = []
    returns_history = []
    states_history.append(old_state)

    while True:
        # step forward
        if np.random.rand() < 0.5:   # left
            new_state = old_state - 1
        else:                        # right
            new_state = old_state + 1

        # append historical states and rewards
        states_history.append(new_state)
        if new_state == left_terminal_state:
            returns_history.append(-1)
        elif new_state == right_terminal_state:
            returns_history.append(1)
        else:
            returns_history.append(0)

        # update state value function
        step += 1
        if step >= n:
            state_values[states_history[step-n]] += alpha * (
                sum(returns_history[step-n:]) + state_values[states_history[-1]] -
                state_values[states_history[step-n]]
            )
        if new_state == right_terminal_state or new_state == left_terminal_state:
            break

        old_state = new_state

    for i in range(step-n+1, step):
        state_values[states_history[i]] += alpha * (
                sum(returns_history[i:]) + state_values[states_history[-1]] -
                state_values[states_history[i]]
        )


alpha = 0.2

r1 = 0
for i in range(100):
    r2 = 0
    state_values = np.zeros(21)
    for j in range(10):
        episode(n=8)
        r2 += mse(state_values)
    r1 += r2 / 10
print(r1 / 100)

# next, loop alpha and episode, we can plot Figure 7.2
