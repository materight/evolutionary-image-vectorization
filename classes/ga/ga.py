from os import replace
import numpy as np
import cv2 as cv
from numpy.random import randint, shuffle, choice

from ..problem import Problem
from .individual import Individual

class GA:
    def __init__(self, target, pop_size=50, n_poly=100, n_vertex=3, selection_cutoff=.1, mutation_chances=(0.01, 0.01, 0.01), mutation_factors=(0.2, 0.2, 0.2), internal_resolution=75):
        self.generation = 0
        self.problem = Problem(Problem.EA, target, internal_resolution)
        self.n_poly = n_poly
        self.n_vertex = n_vertex
        self.selection_cutoff = selection_cutoff
        self.mutation_chances = mutation_chances
        self.mutation_factors = mutation_factors
        self.population = []
        for i in range(pop_size):
            self.population.append(Individual.random(self.problem, self.n_poly, self.n_vertex))
        self.population.sort(key=lambda i: i.fitness)

    def next(self):
        self.generation += 1

        # Tournament selection. Note: the population is already sorted
        selection_count = max(int(len(self.population) * self.selection_cutoff), 2)
        selected = self.population[0:selection_count]
        #shuffle(self.population)
        #selected = [min(self.population[group::selection_count], key=lambda i: i.fitness) for group in range(selection_count)]

        # Crossover
        offspring = []
        for i in range(0, len(self.population)):
        #for i in range(len(selected), len(self.population)):
            p1 = i % selection_count
            p2 = p1
            while p2 == p1:
                p2 = randint(0, selection_count)
            newind = Individual.crossover(selected[p1], selected[p2])
            offspring.append(newind)

        # Mutation
        for ind in offspring:
            ind.mutate(self.mutation_chances, self.mutation_factors)

        self.population = offspring
        #self.population = selected + offspring

        # Return best individual
        self.population.sort(key=lambda i: i.fitness)
        return self.generation, self.population[0], self.population


    def update_target(self, target):
        self.problem.set_target(target)