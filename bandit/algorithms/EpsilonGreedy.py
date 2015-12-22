from math import log, sqrt
from pandas import DataFrame
from random import random, choice

from BaseBanditAlgorithm import BaseBanditAlgorithm

class EpsilonGreedy(BaseBanditAlgorithm):

    """
    Implementation of the epsilon-greedy algorithm for multi-armed bandit testing.
    """

    def __init__(self, epsilon=0.1, counts=[], values=[]):

        """
        Algorithm is controlled by a exploration parameter epsilon.

        The probability of the algorithm choosing the current best arm is
        (1 - epsilon). Otherwise, it will choose between all arms with equal
        probability.

        Inputs:
        epsilon: float -- exploration parameter
        counts: List[int] -- Initial counts for each arm
        values: List[float] -- Initial average reward for each arm
        """

        self.epsilon = epsilon
        self.arms = DataFrame({'Iteration':counts, 'Reward':values})
        self.arms.index.name = 'Arm'
        return

    def initialize(self, n_arms):

        self.arms = DataFrame({'Iteration':[0], 'Reward':[0.0]}, range(n_arms))
        self.arms.index.name = 'Arm'
        return

    def select_arm(self):

        if random() > self.epsilon:
            return self.arms['Reward'].idxmax()
        else:
            return choice(self.arms.index)

    def update(self, chosen_arm, reward):

        arm = int(chosen_arm)
        n = self.arms.ix[arm, 'Iteration'] + 1

        self.arms.ix[arm, 'Iteration'] = n
        self.arms.ix[arm, 'Reward'] *= (n-1)/float(n)
        self.arms.ix[arm, 'Reward'] += reward/float(n)
        return
