import numpy as np
import random
from Easy21 import Easy21
from copy import deepcopy

HIT = 0
STICK = 1


class TDLambda:
    def __init__(self, lmbda=0.0):
        self.Q = np.random.randn(10, 22, 2)
        self.N = np.zeros([10, 22, 2])
        self.epsilon = 1.
        self.lmbda = lmbda
        self.gamma = 0.7

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
        E = np.zeros([10, 22, 2])
        action = self.choose_action(game.state)
        old_state = deepcopy(game.state)
        reward = 0

        while not game.state.game_over:
            reward = game.step(action)
            new_action = self.choose_action(old_state)

            actual_total = old_state.player_total - 1
            if old_state.player_total < 1 or old_state.player_total > 21:
                actual_total = 21
            new_total = game.state.player_total - 1
            if game.state.player_total < 1 or game.state.player_total > 21:
                new_total = 21

            delta = reward + \
                self.gamma * self.Q[game.state.dealers_first_card - 1][new_total][new_action] - \
                self.Q[old_state.dealers_first_card - 1][actual_total][action]

            E[old_state.dealers_first_card - 1][actual_total][action] += 1
            self.N[old_state.dealers_first_card - 1][actual_total][action] += 1
            alpha = 1. / self.N[old_state.dealers_first_card - 1][actual_total][action]

            self.Q = self.Q + alpha * delta * E
            E = self.gamma * self.lmbda * E

            self.epsilon = 100. / (100 + self.N[old_state.dealers_first_card - 1][actual_total][action])
            old_state = deepcopy(game.state)
            action = new_action
        return reward

    def run_episodes(self, n):
        for k in range(1, n+1):
            self.run_episode()

        return self.Q, self.N
