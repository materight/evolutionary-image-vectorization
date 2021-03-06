import numpy as np
from numpy.random import rand, randint, normal, uniform

from .line import Line
from ..operators import velocity_update, topology
from ..utils import sample_points, compute_line_coords

FITNESS_POINTS = 5  # How many points extract from each particle line to compute fitness


class Particle:

    def __init__(self, problem, idx, line, velocity):
        self.problem = problem
        self.idx = idx
        self.line = line
        self.velocity = velocity
        self._fitness = None
        self.personal_best, self.personal_best_fitness = self.line.copy(), self.fitness

    def random(problem, idx, line_length, max_velocity):
        # Init random particle
        line = Line.random(problem, line_length)
        velocity = (rand(line.size) * max_velocity/2) - max_velocity/4  # Random values between (-max/4, +max/4)
        return Particle(problem, idx, line, velocity)

    def move(self, i, swarm, velocity_update_rule, neighborhood_topology, neighborhood_size, coeffs, min_distance, max_velocity):
        # Compute neighborhood.
        # Distance-based
        if isinstance(neighborhood_topology, topology.DistanceTopology):
            sorted_swarm = sorted(swarm, key=lambda p: p.line.dist(self.line))
            neighborhood = sorted_swarm[1:neighborhood_size+1]
        # Ring
        elif isinstance(neighborhood_topology, topology.RingTopology):
            neighborhood_idx = np.arange(i-neighborhood_size//2, i+neighborhood_size//2+1)  # Ring topology
            neighborhood = np.take(swarm, neighborhood_idx, mode='wrap').tolist()
        # Star
        elif isinstance(neighborhood_topology, topology.StarTopology):
            neighborhood_start = (i//neighborhood_size)*neighborhood_size  # Star topolgy
            neighborhood_idx = np.arange(neighborhood_start, neighborhood_start+neighborhood_size)
            neighborhood = np.take(swarm, neighborhood_idx, mode='wrap').tolist()
        else:
            raise ValueError(f'Invalid neighborhood topology "{neighborhood_topology}"')

        # Update velocity
        w, phi1, phi2 = coeffs  # Inertia, cognitive coeff, social coeff
        # Standard PSO
        if isinstance(velocity_update_rule, velocity_update.Standard):
            nhood_best = max(neighborhood, key=lambda p: p.fitness).line if len(neighborhood) > 0 else None
            inertia = w * self.velocity
            cognitive_update = phi1 * rand(self.line.size) * self.personal_best.diff(self.line)
            social_update = phi2 * rand(self.line.size) * nhood_best.diff(self.line) if nhood_best is not None else 0
            self.velocity = inertia + cognitive_update + social_update
        # Fully informed PSO from "Fully Informed Particle Swarm Optimizer: Convergence Analysis"
        elif isinstance(velocity_update_rule, velocity_update.FullyInformed):
            fully_informed = np.zeros_like(self.line.x)
            total_fitness, total_normalization = np.sum([p.fitness for p in neighborhood]), 1e-6
            for p in neighborhood:
                weight = (phi1 + phi2) * (1 / len(neighborhood)) * rand(self.line.size) * (total_fitness - p.fitness)
                total_normalization += weight
                fully_informed += (weight * p.personal_best.diff(self.line))  # Weighted average
            inertia = w * self.velocity
            fully_informed_update = fully_informed / total_normalization
            self.velocity = inertia + fully_informed_update
        # Comprehensive learning PSO
        elif isinstance(velocity_update_rule, velocity_update.ComprehensiveLearning):
            Pc = 0.5
            flb = np.zeros_like(self.line.x)  # Functional local best
            for i in range(len(flb)):
                if rand() > Pc:
                    flb[i] = self.personal_best.x[i]
                else:
                    p1, p2 = np.random.choice(swarm, size=2, replace=False)
                    flb[i] = p1.personal_best.x[i] if p1.personal_best_fitness > p2.personal_best_fitness else p2.personal_best.x[i]
            inertia = w * self.velocity
            flb_update = (phi1 + phi2) * rand(self.line.size) * Line(self.line.img_size, self.line.length, flb).diff(self.line)
            self.velocity = inertia + flb_update
        else:
            raise ValueError(f'Invalid velocity update rule "{velocity_update_rule}"')

        # Maintain a separation between particles in the neighborhood
        for p in swarm:
            if p != self and self.line.dist(p.line) < min_distance:
                self.velocity[:2] -= (p.line.center - self.line.center)

        # Velocity clamping
        if max_velocity > 0:
            self.velocity[self.velocity > max_velocity] = max_velocity
            self.velocity[self.velocity < -max_velocity] = -max_velocity

        # Update line
        self.line.update(self.velocity)

        # Reset fitness
        self._fitness = None

        # Update personal best
        if self.fitness > self.personal_best_fitness:
            self.personal_best, self.personal_best_fitness = self.line.copy(), self.fitness

    @property
    def fitness(self):
        if self._fitness is None:
            pointsL, pointsR = sample_points(self.line.coordsL, FITNESS_POINTS), sample_points(self.line.coordsR, FITNESS_POINTS)
            pointsL, pointsR = np.rint(pointsL).astype(np.int), np.rint(pointsR).astype(np.int)
            sumL = np.sum(self.problem.target[tuple(pointsL.T)].astype(np.int))
            sumR = np.sum(self.problem.target[tuple(pointsR.T)].astype(np.int))
            self._fitness = np.abs(sumR - sumL)  # Image gradient
        return self._fitness
