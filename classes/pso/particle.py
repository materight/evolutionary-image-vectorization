import numpy as np
from numpy.random import rand, randint

from .line import Line

NHOOD_SIZE = 10

class Particle:
    def __init__(self, problem, line, velocity):
        self.problem = problem
        self.line = line
        self.velocity = velocity
        self.personal_best = line
        self._fitness = None

    def random(problem):
        # Init random particle
        line = Line.random(problem)
        velocity = randint(0, 10, line.x.size)
        return Particle(problem, line, velocity)

    def move(self, swarm):
        # Compute neighbor particles
        swarm.sort(key=lambda p: np.abs(self.line.diff(p.line).sum()))
        neighborhood = swarm[:NHOOD_SIZE]
        neighborhood.sort(key=lambda p: p.fitness) # Asynchronous update
        nhood_best = neighborhood[0].line
        # Update velocity
        self.velocity = 1 * self.velocity + 0.7 * rand() * self.personal_best.diff(self.line) + 0.7 * rand() * nhood_best.diff(self.line)
        # Update line 
        self.line = self.line.update(self.velocity)


    @property
    def fitness(self):
        if self._fitness is None:
            self._fitness = 0 # TODO
        return self._fitness