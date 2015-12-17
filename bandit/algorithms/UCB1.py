from math import log, sqrt
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

        ucb_values = self.arms['Reward'] + self.arms['Iteration'].map(
                lambda x: sqrt(2 * log(total_count)/float(x))
        )

        return ucb_values.idxmax()

    def update(self, chosen_arm, reward):

        n = self.update_count(chosen_arm)
        self.update_mean(chosen_arm, reward, n)
        return

    def update_count(self, chosen_arm):

        chosen_arm = int(chosen_arm)

        self.arms.ix[chosen_arm, 'Iteration'] += 1
        return self.arms.ix[chosen_arm, 'Iteration']

    def update_mean(self, chosen_arm, reward, n=None):

        chosen_arm = int(chosen_arm)

        if n == None:
            n = self.arms.ix[chosen_arm, 'Iteration']

        self.arms.ix[chosen_arm, 'Reward'] *= (n-1)/float(n)
        self.arms.ix[chosen_arm, 'Reward'] += reward/float(n)
        return
