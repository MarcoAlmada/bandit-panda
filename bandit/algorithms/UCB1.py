from math import log
from pandas import DataFrame

from BaseBanditAlgorithm import BaseBanditAlgorithm


class UCB1(BaseBanditAlgorithm):

    """
    Implementation of the UCB1 algorithm for multi-armed bandit testing.
    """

    def __init__(self, counts=[], values=[]):

        """
        Algorithm requires no control parameters.

        Inputs:
        counts: List[int] -- Initial counts for each arm
        values: List[float] -- Initial average reward for each arm
        """

        self.arms = DataFrame({'Iteration':counts, 'Reward':values})
        self.arms.index.name = 'Arm'
        return

    def initialize(self, n_arms):

        self.arms = DataFrame({'Iteration':[0], 'Reward':[0.0]}, range(n_arms))
        self.arms.index.name = 'Arm'
        return

    def select_arm(self):

        for arm in self.arms.index:
            if self.arms.ix[arm, 'Iteration'] == 0:
                return arm

        total_count = self.arms['Iteration'].sum()

        ucb_values = 2 * log(total_count)/self.arms['Iteration']
        ucb_values **= 0.5
        ucb_values += self.arms['Reward']

        return ucb_values.idxmax()

    def update(self, chosen_arm, reward):

        arm = int(chosen_arm)
        n = self.arms.ix[arm, 'Iteration'] + 1

        self.arms.ix[arm, 'Iteration'] = n
        self.arms.ix[arm, 'Reward'] *= (n-1)/float(n)
        self.arms.ix[arm, 'Reward'] += reward/float(n)
        return

