# Bandit Panda

This repository is a collection of `pandas`-based implementations of algorithms for the Multi-Armed
Bandit Problem, heavily inspired by John Myles White's _Bandit Algorithms for Website Optimization_.
The original intent of this repository was to implement the algorithms present on the book
(epsilon-greedy, Softmax with and without annealing and UCB1) using the convenient `DataFrame`
structures and functions provided by `pandas`.
However, since it is a "training wheels" project, the repository will probably be expanded with other
algorithms and tools.

# Structure

The `simulation_test.py` script implements Myles White's testing framework:
it creates `Arm` objects based on a list of arm parameters, and then runs a fixed number of
simulations with a predefined run length.
For each simulation, the counts and average rewards for each arm are updated, with the results
of each iteration inside of a simulation (which arm was pulled and what was the reward) being saved
on a `DataFrame` which is then saved as a CSV file.

Implementations of distributions are done through inheritance from the `Arm` class.
Each arm must have a `draw()` method which takes no arguments and returns a single instance of
the reward from that arm.

Algorithms are implemented through inheritance from the `Algorithm` class.
Besides their constructor, they should provide an `update` method which takens as input the ID
of the chosen arm on a given run and the reward from pulling that arm, a `select_arm` method
with no parameters which returns the chosen arm in a given pull, and an `initialize` method
which sets initial values for the counts and average rewards for each arm.

# TO DO

* Write a decent and useful README

* Add other examples of arms

* Improve algorithm performance

* Compare pandas implementation with implementations using other languages or libraries

* Implement Thompson sampling

* Create scripts for test result visualization

* Implement the Environment abstraction from
[Myles White's repository](https://github.com/johnmyleswhite/BanditsBook)

* Introduce sample techniques for result analyses
