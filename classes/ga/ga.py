from os import replace
import numpy as np
import cv2 as cv
from numpy.random import randint, shuffle, choice

from .. import selection
from ..problem import Problem
from .individual import Individual

class GA:
    
    def __init__(self, target, pop_size=50, n_poly=100, n_vertex=3, selection_strategy=selection.TruncatedSelection(0.1), replacement_strategy=selection.CommaStrategy(), crossover_type=Individual.UNIFORM_CROSSOVER, evolution_strategies=False, mutation_chances=(0.01, 0.01, 0.01), mutation_factors=(0.2, 0.2, 0.2), niche_size=0.1, internal_resolution=75):
        self.generation = 0
        self.problem = Problem(Problem.RGB, target, internal_resolution)
        self.pop_size = pop_size
        self.n_poly = n_poly
        self.n_vertex = n_vertex
        self.selection_strategy = selection_strategy
        self.replacement_strategy = replacement_strategy
        self.crossover_type = crossover_type
        self.evolution_strategies = evolution_strategies
        self.mutation_chances = mutation_chances
        self.mutation_factors = mutation_factors
        self.niche_size = niche_size
        self.next_idx = 0
        self.population = []
        for i in range(pop_size):
            self.population.append(Individual.random(self.problem, self.next_idx, self.n_poly, self.n_vertex, self.evolution_strategies))
            self.next_idx += self.n_poly
        self.sort_population()


    def next(self):
        self.generation += 1
        
        # Selection
        selection_probs = np.ones(len(self.population))
        if type(self.selection_strategy) is selection.RouletteWheelSelection: # Roulette wheel selection (fitness-proportionate)
            selection_probs = np.array([i.fitness for i in self.population])
        elif type(self.selection_strategy) is selection.RankBasedSelection: # Rank-based selection
            selection_probs = np.arange(len(self.population), 0, -1)
        elif type(self.selection_strategy) is selection.TruncatedSelection: # Truncated rank-based selection
            selection_count = max(int(len(self.population) * self.selection_strategy.selection_cutoff), 2)
            selection_probs = np.array([1 if i < selection_count else 0 for i in range(len(self.population))])
        elif type(self.selection_strategy) is selection.TournamentSelection: # Tournament selection
            pass # Implemented later when seelction individuals for crossover
        else:
            raise ValueError(f'Invalid selection strategy "{self.selection_strategy}"')
        selection_probs = selection_probs / selection_probs.sum()

        # Crossover
        offspring = []
        for i in range(0, self.pop_size):
            if self.evolution_strategies:
                p1, p2 = self.population[i % len(self.population)], self.population[i % len(self.population)]
            else:
                if type(self.selection_strategy) is selection.TournamentSelection:
                    tournament = np.random.choice(self.population, size=self.selection_strategy.k*2, replace=False)
                    p1, p2 = min(tournament[0::2], key=lambda p: p.fitness), min(tournament[1::2], key=lambda p: p.fitness) # Disjointed tournaments
                else:
                    p1, p2 = np.random.choice(self.population, p=selection_probs, size=2, replace=False)
                newind = Individual.crossover(p1, p2, self.crossover_type)
                offspring.append(newind)

        # Mutation
        for ind in offspring:
            self.next_idx = ind.mutate(self.next_idx, self.mutation_chances, self.mutation_factors)

        # Replace old population
        if type(self.replacement_strategy) is selection.CommaStrategy:
            self.population = offspring
        elif type(self.replacement_strategy) is selection.PlusStrategy:
            self.population = self.population + offspring
        else:
            raise ValueError(f'Invalid replacement strategy "{self.replacement_strategy}"')

        # Sort population by fitness
        pairwise_dist = self.sort_population()
        diversity = pairwise_dist.sum() / 2 if pairwise_dist is not None else None

        # Survivor selection
        self.population = self.population[:self.pop_size]

        return self.generation, self.population, diversity


    def sort_population(self):
        dist = None
        if self.niche_size > 0:
            # Compute pairwise distances
            dist = np.zeros((len(self.population), len(self.population)))
            for i in range(len(self.population)):
                for j in range(i+1, len(self.population)):
                    dist[i, j] = self.population[i].dist(self.population[j])          
                    dist[j, i] = dist[i, j]
            # Compute niche count for each individual
            for i, ind in enumerate(self.population):
                ind.niche_count = (1 - dist[i][dist[i] < self.niche_size] / self.niche_size).sum()
            # Sort population
            self.population.sort(key=lambda i: i.fitness * i.niche_count)
        else:
            self.population.sort(key=lambda i: i.fitness)
        return dist


    def update_target(self, target):
        self.problem.set_target(target)
        