
import numpy as np
import matplotlib.pyplot as plt

# termination state - A - B - C - D - E - termination state
#           0         1   2   3   4   5         6
#                           start
state_set = list(range(0, 7))
state_val_real = [1/6, 2/6, 3/6, 4/6, 5/6]


def random_walk():

    prob = 0.5
    # np.random.rand(1)[0]
    # episodes = []
    # episodes[0] = state_set
    state_list, reward_list = [], []
    start_state = 3
    state_list.append(start_state)

    while True:
        dice = np.random.uniform()
        if dice < prob:
            next_state = start_state + 1
        else:
            next_state = start_state - 1

        state_list.append(next_state)
        if next_state == 0:
            reward_list.append(0)
            break
        if next_state == 6:
            reward_list.append(1)
            break
        reward_list.append(0)
        start_state = next_state

    return state_list, reward_list


def mse(state_val):
    return np.sqrt(np.mean(np.square(np.array(state_val) -
                                     np.array(state_val_real))))


# Monte Carlo Methods
n_episodes = 1000
state_val_est = np.zeros(len(state_set))
state_val_num = np.zeros(len(state_set))

for i in range(n_episodes):
    states, rewards = random_walk()
    for state in states[:-1]:
        state_val_est[state] += rewards[-1]
        state_val_num[state] += 1
print((state_val_est[1: -1] / state_val_num[1: -1]).round(4))
print(np.array(state_val_real).round(4))

plt.plot(state_val_est[1: -1] / state_val_num[1: -1], label='TD')
plt.plot(state_val_real, label='True')
plt.legend()
plt.show()

# Monte Carlo Methods (version 2)
n_episodes = 100
alpha = 0.1
state_val_est = np.zeros(len(state_set))
state_val_num = np.zeros(len(state_set))

for i in range(n_episodes):
    states, rewards = random_walk()
    for state in states[:-1]:
        state_val_num[state] += 1
        # state_val_est[state] = state_val_est[state] + 1 / state_val_num[state] * (
        #         rewards[-1] - state_val_est[state])
        state_val_est[state] = state_val_est[state] + alpha * (
                rewards[-1] - state_val_est[state])

print(state_val_est[1: -1].round(4))
print(np.array(state_val_real).round(4))

plt.plot(state_val_est[1: -1], label='TD')
plt.plot(state_val_real, label='True')
plt.legend()
plt.show()


# TD
# Example 6.2 Random Walk (left figure)
n_episodes = 100
alpha = 0.1
gamma = 1
state_val_est = np.zeros(len(state_set))
state_val_est[1:-1] = 0.5

plt.plot(state_val_est[1:-1], label='0')

for i in range(n_episodes):
    states, rewards = random_walk()
    for j in range(len(states) - 1):
        state_val_est[states[j]] = state_val_est[states[j]] + alpha * (
                rewards[j] + gamma * state_val_est[states[j+1]] - state_val_est[states[j]])
    if i+1 in [1, 10, 100]:
        plt.plot(state_val_est[1:-1], label=str(i+1))

plt.plot(state_val_real, label='True')
plt.legend()
plt.xlabel('State')
plt.ylabel('Estimated Value')
plt.show()

print(state_val_est[1: -1])
print(np.array(state_val_real).round(4))


# Example 6.2 Random Walk (right figure)
n_episodes = 100
alphas = [0.15, 0.1, 0.05]
gamma = 1

for alpha in alphas:
    state_val_est = np.zeros(len(state_set))
    state_val_est[1:-1] = 0.5
    rms = []
    for i in range(n_episodes):
        states, rewards = random_walk()
        for j in range(len(states) - 1):
            state_val_est[states[j]] = state_val_est[states[j]] + alpha * (
                    rewards[j] + gamma * state_val_est[states[j+1]] - state_val_est[states[j]])
        rms.append(mse(state_val_est[1:-1]))
    plt.plot(rms, label=str(alpha))

plt.xlabel('Walks/Episodes')
plt.ylabel('Empirical RMS error, averaged over states')
plt.legend()
plt.grid()
plt.show()
