from GaussianArm import GaussianArm
from BernoulliArm import BernoulliArm
from BaseBanditArm import BaseBanditArm

"""
Arm for modelling a customer's purchasing behaviour.

The model consists on a binomial distribution to find whether or not a given
client makes a purchase and a gaussian distribution from which the rewards
are drawn. 
The arm's output is obtaining by multiplying the output of both distributions.
"""

class RPVArm(BaseBanditArm):

    def __init__(self, p, mean, sigma):

        """Inputs:
        p : float -- probability of making a purchase
        mean : float -- average ticket
        sigma: float -- standard deviation of the average ticket
        """
        
        self.purchase = BernoulliArm(p)
        self.ticket = GaussianArm(mean, sigma)
        return

    def draw(self):

        return self.purchase.draw() * max(self.ticket.draw(), 0)
