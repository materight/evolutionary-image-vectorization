from os import replace
import numpy as np
import cv2 as cv
from numpy.random import randint, shuffle, choice

from .individual import Individual

class Population():
    def __init__(self, target, pop_size=50, n_poly=100, n_vertex=3, selection_cutoff=.1, internal_res=75):
        self.generation = 0
        self.scale_factor = internal_res / min(target.shape[:2])
        self.target = cv.resize(target, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        print('Internal img size: ', self.target.shape)
        self.n_poly = n_poly
        self.n_vertex = n_vertex
        self.selection_cutoff = selection_cutoff
        self.population = []
        for i in range(pop_size):
            self.population.append(Individual.random(self.target, self.scale_factor, self.n_poly, self.n_vertex))
        self.population.sort(key=lambda i: i.fitness)

    def next(self):
        self.generation += 1
        
        # Tournament selection
        selection_count = max(int(len(self.population) * self.selection_cutoff), 2)
        #self.population.sort(key=lambda i: i.fitness)
        #selected = self.population[0:selection_count]
        shuffle(self.population)
        selected = [min(self.population[group::selection_count], key=lambda i: i.fitness) for group in range(selection_count)]
        
        # Crossover
        offspring = []
        for i in range(0, len(self.population)):
        # for i in range(len(selected), len(self.population)):
            p1 = i % selection_count
            p2 = p1
            while p2 == p1: p2 = randint(0, selection_count)
            newind = Individual.crossover(selected[p1], selected[p2])
            offspring.append(newind)
        
        # Mutation
        for ind in offspring:
            ind.mutate()

        self.population = offspring
        #self.population = selected + offspring

        # Return best individual
        best = min(self.population, key=lambda i: i.fitness)
        return self.generation, best, selected