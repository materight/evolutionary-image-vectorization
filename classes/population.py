import numpy as np
import cv2 as cv

from .individual import Individual

class Population():
    def __init__(self, target, pop_size=50, n_poly=100, n_vertex=3, selection_cutoff=.1):
        self.generation = 0
        self.selection_cutoff = selection_cutoff
        self.population = []
        for i in range(pop_size):
            self.population.append(Individual(target, n_poly, n_vertex))

    def next(self):
        self.generation += 1
        self.population.sort(key=lambda i: i.fitness(), reverse=True)

        # Selection
        selected = self.population[0:max(int(len(self.population) * self.selection_cutoff), 1)]

        # Crossover
        offspring = []
        for i in range(len(selected), len(self.population)):
            offspring.append(Individual.crossover(
                np.random.choice(selected),
                np.random.choice(selected)
            ))

        # Mutation
        self.population = selected + offspring
        for i in self.population:
            i.mutate()

        # Return best individual
        self.population.sort(key=lambda i: i.fitness(), reverse=True)
        return self.generation, self.population[0]