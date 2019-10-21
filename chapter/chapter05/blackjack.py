
import numpy as np
from random import choice
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dealer_cards = list(range(1, 11))
player_cards = list(range(12, 22))

return_sets = defaultdict(list)
value_sets = {}


# get a new card
def get_card():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card


def dealer_sim():
    dealer_card = [choice(dealer_cards)]
    while True:
        dealer_card.append(get_card())
        if 17 <= sum(dealer_card) <= 21:
            return sum(dealer_card), dealer_card
        # if dealer has aces
        if dealer_card.count(1) > 0:
            if 17 <= sum(dealer_card) + 10 <= 21:
                return sum(dealer_card) + 10, dealer_card
        if sum(dealer_card) > 21:
            return -1, dealer_card


def player_sim(ace=False):
    player_card = [choice(player_cards)]
    while True:
        if ace:
            pass
        else:
            if 20 <= sum(player_card) <= 21:
                return sum(player_card), player_card
            if sum(player_card) > 21:
                return -1, player_card
        player_card.append(get_card())


for _ in range(500000):

    player_score, player_card = player_sim()
    player_state = np.cumsum(player_card)
    dealer_score, dealer_card = dealer_sim()

    if player_score == -1:
        for s in player_state:
            if 12 <= s <= 21:
                return_sets[(s, dealer_card[0])].append(-1)
    else:
        if player_score > dealer_score:
            for s in player_state:
                if 12 <= s <= 21:
                    return_sets[(s, dealer_card[0])].append(1)
        elif player_score < dealer_score:
            for s in player_state:
                if 12 <= s <= 21:
                    return_sets[(s, dealer_card[0])].append(-1)
        else:
            for s in player_state:
                if 12 <= s <= 21:
                    return_sets[(s, dealer_card[0])].append(0)

value_sets = {state: np.mean(return_sets.get(state)) for state in return_sets}

value_matrix = np.zeros([10, 10])
for i in range(12, 22):
    for j in range(1, 11):
        value_matrix[i-12][j-1] = value_sets.get((i, j))

figure = plt.figure()
figure.set_tight_layout(True)
ax = Axes3D(figure)
x, y = np.meshgrid(np.arange(1, 11), np.arange(12, 22))
ax.plot_surface(x, y, value_matrix)
plt.show()

print(value_matrix.round(1))
