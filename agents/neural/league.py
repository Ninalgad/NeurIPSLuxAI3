import typing
import numpy as np
import math


class Entry(typing.NamedTuple):
    rating: int
    entity: object


class League:
    """
    Prioritized Fictitious self-play
    from: `Grandmaster level in StarCraft II using multi-agent reinforcement learning'

    here, we only save the top k models
    """
    def __init__(self, k):
        self.k = k
        self.entries = []

    def random_matchmaking(self):
        i, j = np.random.choice(len(self.entries), size=2, replace=False)
        return i, j, self.entries[i].entity, self.entries[j].entity

    def preferential_matchmaking(self, rating):
        match_weights = np.array([win_probability(rating, x.rating) for x in self.entries], 'float32')
        match_weights = match_weights * (1 - match_weights)
        match_probs = match_weights / match_weights.sum()
        idx = np.random.choice(len(self.entries), p=match_probs)
        opp = self.entries[idx]
        return idx, opp.entity, opp.rating

    def add(self, init_rating: int, entity: object):
        removed = []
        while len(self.entries) >= self.k:
            worst_idx = np.argmin([ent.rating for ent in self.entries])
            removed.append(self.entries[worst_idx].entity)
            del self.entries[worst_idx]

        self.entries.append(Entry(init_rating, entity))
        return removed

    def update(self, win_index: int, loss_index: int):
        win_ent, loss_ent = self.entries[win_index], self.entries[loss_index]
        r_win, r_loss = elo_rating(win_ent.rating, loss_ent.rating, 1)
        r_win, r_loss = int(np.rint(r_win)), int(np.rint(r_loss))
        self.entries[win_index] = win_ent._replace(rating=r_win)
        self.entries[loss_index] = loss_ent._replace(rating=r_loss)

    def update_rating(self, ent_index: int, new_rating: int):
        self.entries[ent_index] = self.entries[ent_index]._replace(rating=new_rating)

    def save(self, filename):
        np.save(filename, self.entries)

    def load(self, filename):
        ckpt = np.load(filename, allow_pickle=True)
        self.entries = []
        for (rating, entity) in ckpt:
            self.entries.append(Entry(rating, entity))


def win_probability(rating_1, rating_2):
    """Probability player with rating_1 beats player with rating_2"""
    diff = rating_1 - rating_2
    return 1.0 / (1 + math.exp(-diff / 400))


def elo_rating(rating_a, rating_b, outcome, k=30):
    """
    Function to update Elo rating

    :param rating_a: rating of player a
    :param rating_b: rating of player b
    :param outcome: outcome of game = 1 for player a win, 0 for player b win, 0.5 for draw
    :param k: changes in rating are scaled between [-k, k]
    :return: tuple of updated ratings
    """
    # Calculate the Winning Probability of player a
    prob_a = win_probability(rating_a, rating_b)

    # Calculate the Winning Probability of player b
    prob_b = win_probability(rating_b, rating_a)

    # Update the Elo Ratings
    rating_a = rating_a + k * (outcome - prob_a)
    rating_b = rating_b + k * ((1 - outcome) - prob_b)

    return rating_a, rating_b
