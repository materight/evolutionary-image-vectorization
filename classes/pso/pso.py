
import numpy as np
import cv2 as cv
from numpy.random import randint, shuffle, choice

from ..problem import Problem
# from .particle import Particle

class PSO:
   def __init__(self, target, pop_size=50, mutation_chances=(0.01, 0.01, 0.01), mutation_factors=(0.2, 0.2, 0.2), internal_resolution=75):
        self.generation = 0
        self.problem = Problem(Problem.PSO, target, internal_resolution)
        self.mutation_chances = mutation_chances
        self.mutation_factors = mutation_factors
        self.population = []
        for i in range(pop_size):
            pass
            #self.population.append(Particle.random(self.problem, 1, 2))
        self.population.sort(key=lambda i: i.fitness)
        
    

