from random import random

from BaseBanditArm import BaseBanditArm

class BernoulliArm(BaseBanditArm):

    def __init__(self, p):

        self.p = p

    def draw(self):

        if random() > self.p:
            return 0.0
        else:
            return 1.0
