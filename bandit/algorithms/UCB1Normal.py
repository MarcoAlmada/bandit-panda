from math import log
from pandas import DataFrame

from BaseBanditAlgorithm import BaseBanditAlgorithm


class UCB1Normal(BaseBanditAlgorithm):

    """
    UCB1-Normal algorithm, used for rewards with Gaussian support.
    """

    def __init__(self, counts=[], values=[], sumsquares=[]):

        """
        Algorithm requires no control parameters.

        Inputs:
        counts: List[int] -- Initial counts for each arm
        values: List[float] -- Initial average reward for each arm
        """

        self.arms = DataFrame({
            'Iteration':counts, 
            'Reward':values,
            'Sum-of-squares':sumsquares
        })
        self.arms.index.name = 'Arm'
        return

    def initialize(self, n_arms):

        self.arms = DataFrame(
            {
                'Iteration':[0], 
                'Reward':[0.0],
                'Sum-of-squares':[0.0]
            }, 
            range(n_arms)
        )
        self.arms.index.name = 'Arm'
        return

    def select_arm(self):

        total_count = 1 + self.arms['Iteration'].sum()

        arm = self.arms['Iteration'].idxmin()

        if self.arms.ix[arm, 'Iteration'] <= 8 * log(total_count):
            return arm

        sq_diff = self.arms['Sum-of-squares'] - self.arms['Iteration'] * self.arms['Reward'] ** 2
        sq_diff = sq_diff/(self.arms['Iteration'] - 1)

        ucb_values = 16 * log(total_count - 1) / self.arms['Iteration']
        ucb_values *= sq_diff
        ucb_values **= 0.5

        ucb_values += self.arms['Reward']

        return ucb_values.idxmax()

    def update(self, chosen_arm, reward):

        arm = int(chosen_arm)
        n = self.arms.ix[arm, 'Iteration'] + 1

        self.arms.ix[arm, 'Iteration'] = n
        self.arms.ix[arm, 'Reward'] *= (n-1)/float(n)
        self.arms.ix[arm, 'Reward'] += reward/float(n)
        self.arms.ix[arm, 'Sum-of-squares'] += reward**2
        return
