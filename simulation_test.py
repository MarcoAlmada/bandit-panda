import pandas as pd

from bandit.arms import BernoulliArm, LogNormalArm, GaussianArm
from bandit.algorithms import UCB1, UCB1Normal, Softmax, EpsilonGreedy


def create_line(line, algo, arms):

    """
    Given an integer, a Bandit Algorithm, and a list of arms, chooses an arm and gets its reward.

    Inputs:
    line: int -- index for DataFrame construction
    algo: BaseBanditAlgorithm -- algorithm to test
    arms: list of BaseBanditArm -- Arm objects that will be tested with algo

    Output:
    (int, int, float) -- (line index, chosen arm, reward drawn from the arm).
    """

    arm = algo.select_arm()
    reward = arms[arm].draw()
    algo.update(arm, reward)
    return (line, arm, reward)


def run_simulation(algo, arms, sim, horizon):

    """
    Runs a fixed-duration Monte Carlo simulation for a given set of arms.

    Inputs:
    algo: BaseBanditAlgorithm -- algorithm to test
    arms: list of BaseBanditArm -- Arm objects that will be tested with algo
    sim: int -- ID of simulation being run
    horizon: int -- number of iterations on this simulation.

    Output:
    pd.DataFrame -- chosen arm and reward for each iteration of the simulation.
    """

    algo.initialize(len(arms))

    aux = range(horizon)
    aux = map(lambda x: create_line(x, algo, arms), aux)

    res = pd.DataFrame.from_records(aux, columns=['Iteration','Arm','Reward'])

    res['Simulation'] = sim

    return res[['Simulation', 'Iteration', 'Arm', 'Reward']]


def test_algorithm(algo, arms, num_sims=1, horizon=1000):

    """
    Runs a given number of Monte Carlo simulations for the arm set.

    Inputs:
    algo: BaseBanditAlgorithm -- algorithm to test
    arms: list of BaseBanditArm -- Arm objects that will be tested with algo
    num_sim: int -- number of simulations that will be run
    horizon: int -- number of iterations on each simulation

    Output:
    pd.DataFrame -- chosen arm and reward for each iteration of each simulation.
    """

    sims = map(lambda x: run_simulation(algo, arms, x, horizon), range(num_sims))
    return pd.concat(sims, ignore_index=True)
    

if __name__ == "__main__":

    # Normal Arms
    params = [(100, 10), (110, 10), (105, 10)]
    arms = map(lambda p: GaussianArm(p[0], p[1]), params)

    n_sims = 10
    n_iter = 10000

    algo = UCB1Normal()
    algo.initialize(len(arms))

    sim = test_algorithm(algo, arms, n_sims, n_iter)

    sim.to_csv('data/normal_ucb1_simulation_test.csv', index=False)
