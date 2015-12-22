from numpy.random import normal

from BaseBanditArm import BaseBanditArm

class GaussianArm(BaseBanditArm):

    def __init__(self, mean, sigma):

        self.mean = mean
        self.sigma = sigma
        return

    def draw(self):

        return normal(self.mean, self.sigma)
