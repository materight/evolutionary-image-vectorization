import numpy as np
from numpy.random import rand, randint, normal, uniform

from .line import Line

FITNESS_POINTS = 5 # How many points extract from each particle line to compute fitness

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
        velocity = rand(line.size) * 2 - 1
        return Particle(problem, line, velocity)

    def move(self, i, swarm, neighborhood_size, coeffs, min_distance):
        # Compute neighbor particles
        #swarm.sort(key=lambda p: p.line.dist(self.line)) # TODO: optimize neighborhood computation
        #neighborhood = swarm[:neighborhood_size]

        neighborhood_idx = np.concatenate((np.arange(i-neighborhood_size//2,i), np.arange(i+1,i+neighborhood_size//2+1))) # Ring topology
        
        #neighborhood_start = (i//neighborhood_size)*neighborhood_size # Star topolgy
        #neighborhood_idx = np.arange(neighborhood_start, neighborhood_start+neighborhood_size)

        neighborhood = np.take(swarm, neighborhood_idx, mode='wrap').tolist() # Take near particles #range(i+1,i+1+neighborhood_size)

        nhood_best = min(neighborhood, key=lambda p: p.fitness).line
        # Update velocity
        w, phi1, phi2 = coeffs # Inertia, cognitive coeff, social coeff
        inertia = w * self.velocity
        cognitive_update = phi1 * rand(self.line.size) * self.personal_best.diff(self.line)
        social_update = phi2 * rand(self.line.size) * nhood_best.diff(self.line)
        self.velocity = inertia + cognitive_update + social_update
        # Mantain a separation between particles in the neighborhood
        for p in neighborhood:
            if self.line.dist(p.line) < min_distance:
                self.velocity[:2] -= (p.line.center - self.line.center)
        # Update line 
        self.line.update(self.velocity)        
        # Reset fitness
        self._fitness = None

    @property
    def fitness(self):
        if self._fitness is None:
            p1j, p1i, p2j, p2i = self.line.coords
            points = [(p1i, p1j), (p2i, p2j)]
            for k in range(1, FITNESS_POINTS - 1):
                newi = p1i + (((p2i - p1i) / (FITNESS_POINTS - 1)) * k)
                newj = p1j + (((p2j - p1j) / (FITNESS_POINTS - 1)) * k)
                points.append((newi, newj))
            points = np.floor(points).astype(np.int)
            self._fitness = np.sum(self.problem.target[points].astype(np.int)**2)
        return self._fitness