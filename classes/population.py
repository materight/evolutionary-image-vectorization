import numpy as np
import cv2 as cv

from .individual import Individual

class Population():
    def __init__(self, target, pop_size=20, n_poly=100, n_vertex=3):
        self.population = []
        for i in range(pop_size):
            self.population.append(Individual(target.shape, n_poly, n_vertex))
        pass

    def evaluate():
        