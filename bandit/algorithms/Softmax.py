from math import exp, log
from random import random
from pandas import DataFrame

from BaseBanditAlgorithm import BaseBanditAlgorithm

class Softmax(BaseBanditAlgorithm):

    """
    Implementation of the Softmax algorithm for Multi-Armed Bandit    
    """

    def __init__(self, temperature=0.1, annealing=False, counts=[], values=[]):

        """
        Constructor for both standard and annealing Softmax.

        Inputs:
        temperature: float -- controls the exploration phase
        annealing: bool -- If True, temperature changes with time.
        counts: List[int] -- Initial counts for each arm
        values: List[float] -- Initial average reward for each arm
        """

        self.arms = DataFrame({'Iteration':counts, 'Reward':values})
        self.arms.index.name = 'Arm'
        self.annealing = annealing
        
        if annealing:
            self.update_temperature()
        else:
            self.temperature = temperature

        return

    def initialize(self, n_arms):

        """Initiates n_arms arms as blank slates."""

        self.arms = DataFrame({'Iteration':[0], 'Reward':[0.0]}, range(n_arms))
        self.arms.index.name = 'Arm'
        return

    def select_arm(self):

        if self.annealing:
            self.update_temperature()

        probs = self.arms['Reward'].map(lambda x: exp(x/self.temperature))
        probs /= float(probs.sum())

        z = random()
        cum_prob = probs.cumsum()

        return cum_prob[cum_prob > z].index[0]

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

    def update_temperature(self):

        t = 1 + self.arms['Iteration'].sum()
        self.temperature = 1/log(t + 0.0000001)
        return
