from os import replace
import numpy as np
import cv2 as cv

from .individual import Individual

class Population():
    def __init__(self, target, pop_size=50, n_poly=100, n_vertex=3, selection_cutoff=.1):
        self.generation = 0
        self.selection_cutoff = selection_cutoff
        self.population = []
        for i in range(pop_size):
            self.population.append(Individual.random(target, n_poly, n_vertex))
        self.population.sort(key=lambda i: i.fitness)

    def next(self):
        self.generation += 1
        
        # Selection
        selected = self.population[0:max(int(len(self.population) * self.selection_cutoff), 2)]

        # Crossover
        offspring = []
        for i in range(0, len(self.population)):
            offspring.append(Individual.crossover(*np.random.choice(selected, size=2, replace=False)))
        
        # Mutation
        for ind in offspring:
            ind.mutate()

        self.population = offspring
        # self.population = selected + offspring

        # Return best individual
        self.population.sort(key=lambda i: i.fitness)
        return self.generation, self.population[0]