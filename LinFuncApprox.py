import numpy as np
import random
from Easy21 import Easy21
from Easy21 import State
from copy import deepcopy

HIT = 0
STICK = 1


class LinFuncApprox:
    def __init__(self, lmbda=0.0):
        self.N = np.zeros([10, 22, 2])
        self.W = np.zeros([36, ])
        self.epsilon = 0.05
        self.gamma = 0.7
        self.lmbda = lmbda

    def choose_action(self, state):
        random_action = random.randint(0, 1)

        action_choice = np.random.choice(['RAND', 'GREEDY'], p=[self.epsilon, 1 - self.epsilon])

        if action_choice == 'RAND':
            return random_action

        if np.dot(self.generate_features(state, HIT), self.W) == \
                np.dot(self.generate_features(state, STICK), self.W):
            return random_action

        if np.dot(self.generate_features(state, HIT), self.W) > \
                np.dot(self.generate_features(state, STICK), self.W):
            return HIT
        else:
            return STICK

    @staticmethod
    def generate_features(state, action):
        features = np.zeros([3, 6, 2])

        if state.dealers_first_card <= 9 and state.player_total <= 18:
            features[(state.dealers_first_card - 1) / 3][(state.player_total - 1) / 3][action] = 1

        if state.dealers_first_card <= 10 and state.player_total <= 18:
            features[(state.dealers_first_card - 2) / 3][(state.player_total - 1) / 3][action] = 1

        if state.dealers_first_card <= 9 and state.player_total <= 21:
            features[(state.dealers_first_card - 1) / 3][(state.player_total - 4) / 3][action] = 1

        if state.dealers_first_card <= 10 and state.player_total <= 21:
            features[(state.dealers_first_card - 2) / 3][(state.player_total - 4) / 3][action] = 1

        return features.reshape((36,))

    def run_episode(self):
        game = Easy21()
        eligibility_trace = np.zeros([36, ])
        action = self.choose_action(game.state)
        old_state = deepcopy(game.state)
        reward = 0
        alpha = 0.01

        while not game.state.game_over:
            reward = game.step(action)
            new_action = self.choose_action(old_state)

            old_q = np.dot(self.generate_features(old_state, action), self.W)
            new_q = np.dot(self.generate_features(game.state, new_action), self.W)

            delta = reward + self.gamma * new_q - old_q
            eligibility_trace = self.gamma * self.lmbda * eligibility_trace + self.generate_features(old_state, action)
            self.W = self.W + alpha * delta * eligibility_trace

            old_state = deepcopy(game.state)
            action = new_action
        return reward

    def run_episodes(self, n):
        total_reward = []
        for k in range(1, n+1):
            total_reward.append(self.run_episode())

    def q_values(self):
        q_values = np.zeros([10, 22, 2])
        for i in range(1, 11):
            for j in range(1, 22):
                for k in range(0, 2):
                    x = self.generate_features(State(i, j), k)
                    q = np.dot(x, self.W)
                    q_values[i-1][j-1][k] = q

        return q_values

    def error(self, qstar):
        error = 0
        for i in range(1, 11):
            for j in range(1, 22):
                for k in range(0, 2):
                    x = self.generate_features(State(i, j), k)
                    q = np.dot(x, self.W)
                    error += q - qstar[i-1][j-1][k]

        return error
