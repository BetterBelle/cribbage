"""Agents that interact with the CribbageGame."""
import random
from os import path
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod


class Player(metaclass=ABCMeta):
    """Abstract Base Class"""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @abstractmethod
    def select_crib_cards(self, hand, crib=None, self_score=None, opp_score=None, training=False):
        """Select cards to place in crib.

        :param hand: list containing the cards in the player's hand
        :return: list of cards to place in crib
        """
        raise NotImplementedError

    @abstractmethod
    def select_card_to_play(self, hand, table, crib):
        """Select next card to play.

        :param hand: list containing the cards in the player's hand
        :param table: list of all cards that have been played so far during the current round (by all players)
        :param crib: list of cards that the player has placed in the crib
        :return: card to play
        """
        raise NotImplementedError


class RandomPlayer(Player):
    """A player that makes random decisions."""

    def select_crib_cards(self, hand, crib=None, self_score=None, opp_score=None, training=False):
        return random.sample(hand, 2)

    def select_card_to_play(self, hand, table, crib):
        return random.choice(hand)

class MLPlayer(Player):

    def __init__(self, name):
        super().__init__(name)
        self._GOAL = 121
        self._discount_factor = 0.95
        self._eps = 0.5
        self._eps_decay_factor = 0.999

        if path.exists("saved_networks/player_model"):
            self.player_model = tf.keras.models.load_model("saved_networks/player_model")
        else:
            input_layer = tf.keras.Input(shape=(315))
            expand_layer = tf.keras.layers.Dense(400, activation='relu')(input_layer)
            half_expansion = tf.keras.layers.Dense(200, activation='relu')(expand_layer)
            fix = tf.keras.layers.Dense(120, activation='relu')(half_expansion)
            collapse = tf.keras.layers.Dense(30, activation='relu')(fix)
            output_layer = tf.keras.layers.Dense(15, activation='linear')(collapse)

            self.player_model = tf.keras.Model(input_layer, output_layer)
            self.player_model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])
            self.player_model.save("saved_networks/player_model")

    def reset_weights(self):
        self.player_model = tf.keras.models.load_model("saved_networks/player_model")

    def train_crib_cards(self, discarded, hand, self_score, opp_score, self_round_score, opp_round_score, crib):
        choice = 0
        if discarded[0] == 0:
            choice = discarded[1] - 1
        elif discarded[0] == 1:
            choice = discarded[1] + 3
        elif discarded[0] == 2:
            choice = discarded[1] + 6
        elif discarded[0] == 3:
            choice = discarded[1] + 8
        elif discarded[0] == 4:
            choice = discarded[1] + 9

        # initialize prediction array
        input_to_model = []
        # convert hand to array
        for card in hand:
            hand_as_bits = [0 for _ in range(52)]
            hand_as_bits[card.rank['rank'] - 1 + (card.suit['rank'] - 1) * 13] = 1
            input_to_model.extend(hand_as_bits)

        # convert self_score and opp_score to percentage
        self_score = (self_score - self_round_score) / self._GOAL
        opp_score = (opp_score - opp_round_score) / self._GOAL

        # append all to an array in order hand -> selfscore -> oppscore -> crib
        input_to_model += [self_score]
        input_to_model += [opp_score]
        input_to_model += [crib]

        prediction = self.player_model.predict(np.array([input_to_model]))[0]
        target = (self_round_score - opp_round_score) + self._discount_factor * np.max(prediction)
        prediction[choice] = target

        
        history = self.player_model.fit(np.array([input_to_model]), np.array([prediction]), epochs=1, verbose=0)
        return history.history['mean_absolute_error']

    def select_crib_cards(self, hand, crib=None, self_score=None, opp_score=None, training=False):
        choices = {
            0: [0, 1],
            1: [0, 2],
            2: [0, 3],
            3: [0, 4],
            4: [0, 5],
            5: [1, 2],
            6: [1, 3],
            7: [1, 4],
            8: [1, 5],
            9: [2, 3],
            10: [2, 4],
            11: [2, 5],
            12: [3, 4],
            13: [3, 5],
            14: [4, 5],
        }
        if training:
            self._eps *= self._eps_decay_factor
            if np.random.random() < self._eps:
                choice = choices[np.random.randint(0, 15)]
                return [hand[choice[0]], hand[choice[1]]]
                
        # initialize prediction array
        input_to_model = []
        # convert hand to array
        
        for card in hand:
            hand_as_bits = [0 for _ in range(52)]
            hand_as_bits[card.rank['rank'] - 1 + (card.suit['rank'] - 1) * 13] = 1
            input_to_model.extend(hand_as_bits)

        # convert self_score and opp_score to percentage
        self_score /= self._GOAL
        opp_score /= self._GOAL

        input_to_model.append(self_score)
        input_to_model.append(opp_score)
        input_to_model.append(crib)

        thing = self.player_model.predict(np.array([input_to_model]))[0]
        cards_to_pick = choices[np.argmax(self.player_model.predict(np.array([input_to_model]))[0])]

        return [hand[cards_to_pick[0]], hand[cards_to_pick[1]]] # predicted value

    def select_card_to_play(self, hand, table, crib):
        return random.choice(hand)


class HumanPlayer(Player):
    """Interface for a human user to play."""

    def present_cards_for_selection(self, cards, n_cards=1):
        """Presents a text-based representation of the game via stdout and prompts a human user for decisions.

        :param cards: list of cards in player's hand
        :param n_cards: number of cards that player must select
        :return: list of n_cards cards selected from player's hand
        """
        cards_selected = []
        while len(cards_selected) < n_cards:
            s = ""
            for idx, card in enumerate(cards):
                s += "(" + str(idx + 1) + ") " + str(card)
                if card != cards[-1]:
                    s += ","
                s += " "
            msg = "Select a card: " if n_cards == 1 else "Select %d cards: " % n_cards
            print(s)
            selection = input(msg)
            card_indices = [int(s) for s in selection.split() if s.isdigit()]
            for idx in card_indices:
                if idx < 1 or idx > len(cards):
                    print("%d is an invalid selection." % idx)
                else:
                    cards_selected.append(cards[idx-1])
        return cards_selected

    def select_crib_cards(self, hand, crib=None, self_score=None, opp_score=None, training=False):
        return self.present_cards_for_selection(cards=hand, n_cards=2)

    def select_card_to_play(self, hand, table, crib):
        return self.present_cards_for_selection(cards=hand, n_cards=1)[0]
