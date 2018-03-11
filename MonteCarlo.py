import numpy as np
import random
from Easy21 import Easy21
from copy import deepcopy

HIT = 0
STICK = 1


class MonteCarlo:
    def __init__(self):
        self.Q = np.random.randn(10, 22, 2)
        self.N = np.zeros([10, 22, 2])
        self.epsilon = 1.

    def choose_action(self, state):
        random_action = random.randint(0, 1)

        action_choice = np.random.choice(['RAND', 'GREEDY'], p=[self.epsilon, 1 - self.epsilon])

        if action_choice == 'RAND':
            return random_action

        actual_total = state.player_total - 1
        if state.player_total < 1 or state.player_total > 21:
            actual_total = 21

        if self.Q[state.dealers_first_card - 1][actual_total][HIT] == \
                self.Q[state.dealers_first_card - 1][actual_total][STICK]:
            return random_action

        if self.Q[state.dealers_first_card - 1][actual_total][HIT] > \
                self.Q[state.dealers_first_card - 1][actual_total][STICK]:
            return HIT
        else:
            return STICK

    def run_episode(self):
        game = Easy21()
        action = self.choose_action(game.state)
        old_state = deepcopy(game.state)
        reward = 0
        new_q = deepcopy(self.Q)
        while not game.state.game_over:
            reward = game.step(action)

            actual_total = old_state.player_total - 1
            if old_state.player_total < 1 or old_state.player_total > 21:
                actual_total = 21

            self.N[old_state.dealers_first_card - 1][actual_total][action] += 1
            alpha = 1. / self.N[old_state.dealers_first_card - 1][actual_total][action]
            new_q[old_state.dealers_first_card - 1][actual_total][action] += \
                alpha * (reward - self.Q[old_state.dealers_first_card - 1][actual_total][action])

            self.epsilon = 100. / (100 + self.N[old_state.dealers_first_card - 1][actual_total][action])
            action = self.choose_action(old_state)
            old_state = deepcopy(game.state)

        self.Q = deepcopy(new_q)
        return reward

    def run_episodes(self, n):
        for k in range(1,n+1):
            self.run_episode()

        return self.Q, self.N
