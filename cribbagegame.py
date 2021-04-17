"""Cribbage game."""
import random
import time
import numpy as np
from matplotlib import pyplot
from cribbage import scoring
from cribbage.player import HumanPlayer, RandomPlayer, MLPlayer
from cribbage.playingcards import Deck

DEBUG = False  # Debug flag for debugging output


def debug(s):
    """Print debug-level output when debugging is enabled."""
    if DEBUG:
        print(s)


class CribbageGame:
    """Main cribbage game class."""
    
    # Class-level constants
    MAX_SCORE = 121  # game ends at this score
    CRIB_SIZE = 4  # size of the crib
    N_GO = 31  # round sequence ends at this card point total

    def __init__(self, players):
        self.players = players  #: the two players
        self.board = CribbageBoard(self.players, self.MAX_SCORE)  #: the cribbage board for scoring
        assert len(self.players) == 2, "Currently, only 2-player games are supported."

    def _alternate_players(self, start_idx=0):
        """Generator to alternate which player dealers, with an arbitrary starting player.

        :param start_idx: Index of player who deals initially.
        :return: Generator primed for the start_idx player to deal.
        """
        if start_idx == 1:
            yield self.players[1]
        while True:
            yield self.players[0]
            yield self.players[1]

    def start(self):
        """Initiate game.

        :return: None
        """
        starting_player = random.choice([0, 1])
        # print("Coin flip. %s is dealer." % str(self.players[starting_player]))
        player_gen = self._alternate_players(starting_player)
        game_score = [0 for _ in self.players]
        player_histories = {p: [] for p in self.players}
        while max(game_score) < self.MAX_SCORE:
            dealer = next(player_gen)
            r = CribbageRound(self, dealer=dealer)
            # Normal game
            # r.play()

            # Training 
            histories = r.train_discarding()
            for player_history in player_histories:
                if dealer == player_history:
                    player_histories[player_history].extend(histories[0])
                else:
                    player_histories[player_history].extend(histories[1])
            game_score = [self.board.get_score(p) for p in self.players]
            debug(self.board)

        return [np.argmax(game_score), player_histories]


class CribbageRound:
    """Individual round of cribbage."""

    def __init__(self, game, dealer):
        # Replenish deck for each round
        self.deck = Deck()
        self.game = game
        self.hands = {player: [] for player in self.game.players}
        self.crib = []
        self.table = []
        self.starter = None
        self.dealer = dealer
        self.nondealer = [p for p in self.game.players if p != dealer][0]

    def _deal(self):
        """Deal cards.

        :return: None
        """
        shuffles = 3  # ACC Rule 2.1
        cards_per_player = 6
        for i in range(shuffles):
            self.deck.shuffle()
        for _ in range(cards_per_player):
            for p in self.game.players:
                self.hands[p].append(self.deck.draw())
        # print("Cards dealt.")

    def _populate_crib(self):
        """Solicit crib card decisions from players and place these cards in the crib.

        :return: None
        """
        for p in self.game.players:
            cards_to_crib = None
            if p == self.dealer:
                cards_to_crib = p.select_crib_cards(self.hands[p], 1, self.game.board.get_score(self.dealer), self.game.board.get_score(self.nondealer), False)
            else:
                cards_to_crib = p.select_crib_cards(self.hands[p], 0, self.game.board.get_score(self.nondealer), self.game.board.get_score(self.dealer), False)
            debug("Cards cribbed: %s" % cards_to_crib)
            if not set(cards_to_crib).issubset(set(self.hands[p])):
                raise IllegalCardChoiceError("Crib cards selected are not part of player's hand.")
            elif len(cards_to_crib) != 2:
                raise IllegalCardChoiceError("Wrong number of cards sent to crib.")
            else:
                self.crib += cards_to_crib
                for card in cards_to_crib:
                    self.hands[p].remove(card)
        assert len(self.crib) == self.game.CRIB_SIZE, "Crib size is not %s" % self.game.CRIB_SIZE

    def table_to_str(self, sequence_start_idx):
        """Render current table state as a string.

        This function converts the current table state into a string consisting of all previous up-to-31 sequences
        in parentheses followed by the current (active) sequence.
        :param sequence_start_idx: Start index of the current sequence.
        :return: String representing the current table state.
        """
        prev, curr = "", ""
        for play in self.table[:sequence_start_idx]:
            prev += str(play['card']) + ", "
        if prev:
            prev = "(" + prev[:-2] + ") "
        for play in self.table[sequence_start_idx:]:
            curr += str(play['card']) + ", "
        return prev + curr[:-2]

    def _cut(self):
        """Cut the deck."""
        cut_point = random.randrange(len(self.deck))
        self.deck.cut(cut_point=cut_point)
        # print("Cards cut.")

    def get_table_value(self, sequence_start_idx):
        """Get the total value of cards in the current active sequence.

        :param sequence_start_idx: Table index where current sequence begins.
        :return: Total value of cards in active sequence.
        """
        return sum(i['card'].get_value() for i in self.table[sequence_start_idx:]) if self.table else 0

    def play(self):
        """Start cribbage round."""
        loser = None
        self._cut()
        self._deal()
        debug(self.hands)
        self._populate_crib()
        self._cut()
        self.starter = self.deck.draw()
        if self.starter.get_rank() == 'jack':
            self.game.board.peg(self.dealer, 1)
            # print("2 points to %s for his heels." % str(self.dealer))
        active_players = [self.nondealer, self.dealer]
        while sum([len(v) for v in self.hands.values()]):
            sequence_start_idx = len(self.table)
            while active_players:
                for p in active_players:
                    # print("Table: " + self.table_to_str(sequence_start_idx))
                    # print("Player %s's hand: %s" % (p, self.hands[p]))
                    card = p.select_card_to_play(hand=self.hands[p], table=self.table[sequence_start_idx:],
                                                 crib=self.crib)
                    if card.get_value() + self.get_table_value(sequence_start_idx) > 31 or card is None:
                        # print("Player %s chooses go." % str(p))
                        loser = loser if loser else p
                        active_players.remove(p)
                        # If no one can play any more cards, give point to player of last card played
                    else:
                        self.table.append({'player': p, 'card': card})
                        self.hands[p].remove(card)
                        if not self.hands[p]:
                            active_players.remove(p)
                        # print("Player %s plays %s for %d" %
                            #   (str(p), str(card), self.get_table_value(sequence_start_idx)))
                        # Consider cards played by both players when scoring during play
                        assert self.get_table_value(sequence_start_idx) <= 31, \
                            "Value of cards on table must be <= 31 to be eligible for scoring."
                        score = self._score_play(card_seq=[move['card'] for move in self.table[sequence_start_idx:]])
                        if score:
                            self.game.board.peg(p, score)
            # If both players have reached 31 or "go" and not run out of cards, continue play
            if len(active_players) == 0:
                player_of_last_card = self.table[-1]['player']
                self.game.board.peg(player_of_last_card, 1)
                # print("Point to %s for last card played." % player_of_last_card)
                active_players = [p for p in self.game.players if p != player_of_last_card and self.hands[p]]
                if self.hands[player_of_last_card]:
                    active_players += [player_of_last_card]

        # Score each player's hand
        for p in self.game.players:
            p_cards_played = [move['card'] for move in self.table if move['player'] == p]
            # print("Scoring " + str(p) + "'s hand: " + str(p_cards_played + [self.starter]))
            #score = self._score_hand(cards=p_cards_played + [self.starter])  # Include starter card as part of hand
            score = self._score_hand(hand = p_cards_played, s_card = self.starter, is_crib = False)
            if score:
                self.game.board.peg(p, score)

        # Score the crib
        # print("Scoring the crib: " + str(self.crib + [self.starter]))
        #score = self._score_hand(cards=(self.crib + [self.starter]))  # Include starter card as part of crib
        score = self._score_hand(hand = self.crib, s_card = self.starter, is_crib = True)
        if score:
            self.game.board.peg(self.dealer, score)

    def train_discarding(self):
        winner = None
        dealer_history = []
        non_dealer_history = []
        self._cut()
        self._deal()
        crib_cards = self._populate_crib_train()
        self._cut()
        self.starter = self.deck.draw()
        dealer_score = 0
        non_dealer_score = 0

        # Score each player's hand
        for p in self.game.players:
            # print("Scoring " + str(p) + "'s hand: " + str(self.hands[p] + [self.starter]))
            #score = self._score_hand(cards=p_cards_played + [self.starter])  # Include starter card as part of hand
            
            score = self._score_hand(hand = self.hands[p], s_card = self.starter, is_crib = False)

            if p == self.dealer:
                dealer_score = score 
            else:
                non_dealer_score = score
            if score:
                self.game.board.peg(p, score)

        # Score the crib
        # print("Scoring the crib: " + str(self.crib + [self.starter]))
        #score = self._score_hand(cards=(self.crib + [self.starter]))  # Include starter card as part of crib
        crib_score = self._score_hand(hand = self.crib, s_card = self.starter, is_crib = True)
        if crib_score:
            self.game.board.peg(self.dealer, crib_score)

        dealer_score += crib_score

        # Train
        if isinstance(self.dealer, MLPlayer):
            self.hands[self.dealer].insert(crib_cards[0][0], crib_cards[0][1])
            self.hands[self.dealer].insert(crib_cards[0][2], crib_cards[0][3])
            dealer_history.extend(self.dealer.train_crib_cards(
                discarded=[crib_cards[0][0], crib_cards[0][2]], 
                hand=self.hands[self.dealer], 
                self_score=self.game.board.get_score(self.dealer), 
                opp_score=self.game.board.get_score(self.nondealer),
                self_round_score=dealer_score, 
                opp_round_score=non_dealer_score,
                crib=1
            ))
        if isinstance(self.nondealer, MLPlayer):
            self.hands[self.nondealer].insert(crib_cards[1][0], crib_cards[1][1])
            self.hands[self.nondealer].insert(crib_cards[1][2], crib_cards[1][3])
            non_dealer_history.extend(self.nondealer.train_crib_cards(
                discarded=[crib_cards[1][0], crib_cards[1][2]], 
                hand=self.hands[self.nondealer],
                self_score=self.game.board.get_score(self.nondealer), 
                opp_score=self.game.board.get_score(self.dealer), 
                self_round_score=non_dealer_score, 
                opp_round_score=dealer_score,
                crib=0
            ))
        
        return [dealer_history, non_dealer_history]

    def _populate_crib_train(self):
        dealer_crib_cards = []
        other_crib_cards = []
        for p in self.game.players:
            cards_to_crib = None
            if p == self.dealer:
                cards_to_crib = p.select_crib_cards(self.hands[p], 1, self.game.board.get_score(self.dealer), self.game.board.get_score(self.nondealer), True)
            else:
                cards_to_crib = p.select_crib_cards(self.hands[p], 0, self.game.board.get_score(self.nondealer), self.game.board.get_score(self.dealer), True)

            debug("Cards cribbed: %s" % cards_to_crib)
            if not set(cards_to_crib).issubset(set(self.hands[p])):
                raise IllegalCardChoiceError("Crib cards selected are not part of player's hand.")
            elif len(cards_to_crib) != 2:
                raise IllegalCardChoiceError("Wrong number of cards sent to crib.")
            else:
                self.crib += cards_to_crib
                if self.dealer == p:
                    for card in cards_to_crib:
                        dealer_crib_cards.append(self.hands[p].index(card))
                        dealer_crib_cards.append(card)
                else:
                    for card in cards_to_crib:
                        other_crib_cards.append(self.hands[p].index(card))
                        other_crib_cards.append(card)

                for card in cards_to_crib:
                    self.hands[p].remove(card)

        assert len(self.crib) == self.game.CRIB_SIZE, "Crib size is not %s" % self.game.CRIB_SIZE

        return [dealer_crib_cards, other_crib_cards]

    def _score_play(self, card_seq):
        """Return score for latest move in an active play sequence.

        :param card_seq: List of all cards played (oldest to newest).
        :return: Points earned by player of latest card played in the sequence.
        """
        score = 0
        score_scenarios = [scoring.ExactlyEqualsN(n=15), scoring.ExactlyEqualsN(n=31),
                           scoring.HasPairTripleQuad_DuringPlay(), scoring.HasStraight_DuringPlay()]
        for scenario in score_scenarios:
            s, desc = scenario.check(card_seq[:])
            score += s
            # print("[SCORE] " + desc) if desc else None
        return score

    def _score_hand(self, hand, s_card, is_crib):
        """Score a hand at the end of a round.

        :param cards: Cards in a single player's hand.
        :return: Points earned by player.
        """
        score = 0
        score_scenarios = [scoring.NinHand(hand, s_card),
                           scoring.HasPairTripleQuad_InHand(hand, s_card), scoring.HasStraight_InHand(hand, s_card), scoring.HasFlush(hand, s_card, is_crib)]
        for scenario in score_scenarios:
            s, desc = scenario.check()
            score += s
            # print("[FOR SCORING] " + desc) if desc else None
        return score


class CribbageBoard:
    """Board used to peg (track) points during a cribbage game."""

    def __init__(self, players, max_score):
        self.max_score = max_score
        self.pegs = {p: {'front': 0, 'rear': 0} for p in players}

    def __str__(self):
        s = "[PEGS] "
        for player in self.pegs.keys():
            s += str(player) + " "
            for peg in self.pegs[player]:
                s += str(peg) + ": " + str(self.pegs[player][peg]) + ", "
            s = s[:-2]
            s += "; "
        s = s[:-2]
        return s

    def __repr__(self):
        return str(self)

    def peg(self, player, points):
        """Add points for a player.

        :param player: Player who should receive points.
        :param points: Number of points to add to the player.
        :return: None
        """
        assert points > 0, "You must peg 1 or more points."
        self.pegs[player]['rear'] = self.pegs[player]['front']
        self.pegs[player]['front'] += points
        if self.pegs[player]['front'] > self.max_score:
            self.pegs[player]['front'] = self.max_score

    def get_score(self, player):
        """Return a given player's current score.

        :param player: Player to check.
        :return: Score for player.
        """
        return self.pegs[player]['front']


class IllegalCardChoiceError(Exception):
    """Raised when player client returns an invalid card selection."""

    pass


def main():
    games_to_play = 2500
    games_per_series = 100
    players = [MLPlayer("Player1"), MLPlayer("Player2")]
    player_histories = {players[0]: [], players[1]: []}
    winning_histories = []
    wins = [0, 0]
    
    start = time.time()
    for i in range(1, games_to_play + 1):
        game = CribbageGame(players=players)
        info = game.start()
        wins[info[0]] += 1
        player_histories[players[0]].append(info[1][players[0]])
        player_histories[players[1]].append(info[1][players[1]])
        if i % games_per_series == 0:
            winning_player = np.argmax(wins)
            players[winning_player].player_model.save("saved_networks/player_model")
            winning_histories.append(player_histories[players[winning_player]])
            player_histories = {players[0]: [], players[1]: []}

            for player in players:
                player.reset_weights()

            print (wins)
            wins = [0, 0]
        

    if wins != [0, 0]:
        winning_player = np.argmax(wins)
        players[winning_player].player_model.save("saved_networks/player_model")
        winning_histories.append(player_histories[players[winning_player]])
        print(wins)

    series_number = 0
    fig, axs = pyplot.subplots(5,5)
    fig.suptitle("Average of Mean Absolute Error of Winner per Game by Series")
    for game_series in winning_histories:
        avgs_per_game = []
        for game in game_series:
            avgs_per_game.append(sum(game) / len(game))

        axs.flat[series_number].set_title('Series ' + str(series_number + 1))
        axs.flat[series_number].plot([n for n in range(1, games_per_series + 1)], avgs_per_game)
        axs.flat[series_number].set(xlabel='Games Played', ylabel='Mean Absolute Error')
        series_number += 1

    pyplot.show()
    print(time.time() - start)


if __name__ == '__main__':
    main()
