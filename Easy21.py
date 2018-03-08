import random
import numpy as np

HIT = 0
STICK = 1


class Easy21:
    def __init__(self):
        self.player = Player()
        self.dealer = Player()
        self.state = State(self.dealer.get_first_card().number, self.player.get_current_total())

    def hit(self):
        self.player.draw_card()

        if self.player.is_bust():
            return -1
        return 0

    def stick(self):
        while self.dealer.current_total < 17 and not self.dealer.is_bust():
            self.dealer.draw_card()

        if self.dealer.is_bust():
            return 1

        if self.dealer.current_total > self.player.current_total:
            return -1
        if self.dealer.current_total < self.player.current_total:
            return 1
        return 0

    def step(self, action):
        if action == HIT:
            reward = self.hit()
            if reward == 0:
                self.update_state()
            else:
                self.update_state(True)

        if action == STICK:
            reward = self.stick()
            self.update_state(True)

        return reward

    def get_current_state(self):
        return self.state

    def update_state(self, game_over=False):
        self.state.dealers_first_card = self.dealer.get_first_card().number
        self.state.player_total = self.player.get_current_total()
        self.state.game_over = game_over


class Player:
    def __init__(self):
        self.current_cards = []
        self.current_total = 0
        self.draw_card(colour='black')

    def draw_card(self, colour=None, number=None):
        actual_colour = colour if colour is not None else np.random.choice(['red', 'black'], p=[1./3, 2./3])
        actual_number = number if number is not None else random.randint(1, 10)

        self.current_cards.append(Card(actual_colour, actual_number))

        if actual_colour == 'black':
            self.current_total += actual_number
        else:
            self.current_total -= actual_number

    def is_bust(self):
        return self.current_total < 1 or self.current_total > 21

    def get_current_total(self):
        return self.current_total

    def get_first_card(self):
        return self.current_cards[0]


class State:
    def __init__(self, dealers_first_card, player_total, game_over=False):
        self.dealers_first_card = dealers_first_card
        self.player_total = player_total
        self.game_over = game_over


class Card:
    def __init__(self, colour, number):
        self.colour = colour
        self.number = number

