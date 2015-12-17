class BaseBanditAlgorithm():

    def initialize(self, n_arms):
        raise NotImplementedError()

    def select_arm(self):
        raise NotImplementedError()

    def update(self, chosen_arm, reward):
        raise NotImplementedError()
