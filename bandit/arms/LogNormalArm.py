from numpy.random import lognormal

from BaseBanditArm import BaseBanditArm

class LogNormalArm(BaseBanditArm):

    def __init__(self, mean, sigma):

        """Mean and deviation refer to the underlying Normal distribution.
        """

        self.mean = mean
        self.sigma = sigma
        return

    def draw(self):

        return lognormal(self.mean, self.sigma)
