import numpy as np
from numpy.random import rand, randint

from .line import Line

NHOOD_SIZE = 5

W = 0.5 # Inertia
PHI1 = 3.1 # Cognitive coeff
PHI2 = 1.1 # Social coeff


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
        velocity = rand(line.x.size) * 2 - 1
        return Particle(problem, line, velocity)

    def move(self, swarm):
        # Compute neighbor particles
        swarm.sort(key=lambda p: np.abs(self.line.diff(p.line).sum())) # TODO: optimiza neighborhood computation
        neighborhood = swarm[:NHOOD_SIZE]
        neighborhood.sort(key=lambda p: p.fitness) # Asynchronous update
        nhood_best = neighborhood[0].line
        # Update velocity
        self.velocity = W * self.velocity + PHI1 * rand() * self.personal_best.diff(self.line) + PHI2 * rand() * nhood_best.diff(self.line)
        # Update line 
        self.line.update(self.velocity)
        # Reset fitness
        self._fitness = None

    @property
    def fitness(self):
        if self._fitness is None:
            j, i = np.floor(self.line.center).astype(np.int)
            self._fitness = self.problem.target[i, j] # TODO: average of all pixels inside line
        return self._fitness