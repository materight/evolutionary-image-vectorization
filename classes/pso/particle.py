import numpy as np
from numpy.random import rand, randint, normal, uniform

from .line import Line

FITNESS_POINTS = 5 # How many points extract from each particle line to compute fitness

class Particle:
    def __init__(self, problem, line, velocity):
        self.problem = problem
        self.line = line
        self.velocity = velocity
        self._fitness = None
        self.personal_best, self.personal_best_fitness = self.line.copy(), self.fitness

    def random(problem, max_velocity):
        # Init random particle
        line = Line.random(problem)
        velocity = (rand(line.size) * max_velocity/2) - max_velocity/4 # Random values between (-max/4, +max/4)
        return Particle(problem, line, velocity)

    def move(self, i, swarm, neighborhood_size, coeffs, min_distance, max_velocity):
        # Compute neighbor particles
        #swarm.sort(key=lambda p: p.line.dist(self.line)) # TODO: optimize neighborhood computation
        #neighborhood = swarm[:neighborhood_size]

        #neighborhood_idx = np.arange(i-neighborhood_size//2, i+neighborhood_size//2+1) # Ring topology
        
        neighborhood_start = (i//neighborhood_size)*neighborhood_size # Star topolgy
        neighborhood_idx = np.arange(neighborhood_start, neighborhood_start+neighborhood_size)

        neighborhood = np.take(swarm, neighborhood_idx, mode='wrap').tolist() # Take near particles

        # Update velocity
        
        # Standard PSO
        w, phi1, phi2 = coeffs # Inertia, cognitive coeff, social coeff
        nhood_best = min(neighborhood, key=lambda p: p.fitness).line
        inertia = w * self.velocity
        cognitive_update = phi1 * rand(self.line.size) * self.personal_best.diff(self.line)
        social_update = phi2 * rand(self.line.size) * nhood_best.diff(self.line)
        self.velocity = inertia + cognitive_update + social_update
        
        # Fully informed PSO from "Fully Informed Particle Swarm Optimizer: Convergence Analysis"
        fully_informed = np.zeros_like(self.line.x)
        total_fitness, total_normalization = np.sum([p.fitness for p in neighborhood]), 0
        for p in neighborhood:
            weight = (phi1 + phi2) * (1 / len(neighborhood)) * rand(self.line.size) * (total_fitness - p.fitness)
            total_normalization += weight
            fully_informed += (weight * p.personal_best.diff(self.line)) # Weighted average
        inertia = w * self.velocity
        fully_informed_update = fully_informed / total_normalization
        self.velocity = inertia + fully_informed_update

        # Comprehensive learning PSO
        Pc = 0.5
        flb = np.zeros_like(self.line.x) # Functional local best
        for i in range(len(flb)):
            if rand() > Pc:
                flb[i] = self.personal_best.x[i]
            else:
                p1, p2 = np.random.choice(swarm, size=2, replace=False)
                flb[i] = p1.personal_best.x[i] if p1.personal_best_fitness > p2.personal_best_fitness else p2.personal_best.x[i]
        inertia = w * self.velocity
        flb_update = (phi1 + phi2) * rand(self.line.size) * Line(self.line.img_size, flb).diff(self.line)
        self.velocity = inertia + flb_update

        # Maintain a separation between particles in the neighborhood
        for p in neighborhood:
            if self.line.dist(p.line) < min_distance:
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
        if self.fitness < self.personal_best_fitness:
            self.personal_best, self.personal_best_fitness = self.line.copy(), self.fitness

    @property
    def fitness(self):
        if self._fitness is None:
            p1j, p1i, p2j, p2i = self.line.coords
            w, h = (p2i - p1i) / FITNESS_POINTS, (p2j - p1j) / FITNESS_POINTS
            points = [(p1i + w * k, p1j + h * k)  for k in range(0, FITNESS_POINTS)]
            points = np.floor(points).astype(int)


            '''            
            import matplotlib.pyplot as plt
            plt.imshow(self.problem.target)
            print(points)
            print(self.problem.target[tuple(points.T)])
            plt.plot(self.line.coords.T[0], self.line.coords.T[1], c='k')
            for p in points:
                plt.scatter(p[1], p[0], c='r', s=1)
            for p in points[:1]:
                print(p, self.problem.target[(p[0]), (p[1])])
                plt.annotate(self.problem.target[(p[0]), (p[1])], (p[1], p[0]))
            plt.show()
            '''
            
            self._fitness = np.sum(self.problem.target[tuple(points.T)].astype(np.int)**2)
        return self._fitness