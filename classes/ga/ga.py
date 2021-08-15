from os import replace
import numpy as np
import cv2 as cv
from numpy.random import randint, random_sample, shuffle, choice

from ..operators import selection, replacement
from ..problem import Problem
from .individual import Individual

class GA:
    

    def __init__(self, target, pop_size=50, n_poly=100, n_vertex=3, selection_strategy=selection.TruncatedSelection(0.1), replacement_strategy=replacement.CommaReplacement(), crossover_type=Individual.UNIFORM_CROSSOVER, self_adaptive=False, mutation_rates=(0.01, 0.01, 0.01), mutation_step_sizes=(0.2, 0.2, 0.2), internal_resolution=75):
        self.generation = 0
        self.problem = Problem(Problem.RGB, target, internal_resolution)
        self.pop_size = pop_size
        self.n_poly = n_poly
        self.n_vertex = n_vertex
        self.selection_strategy = selection_strategy
        self.replacement_strategy = replacement_strategy
        self.crossover_type = crossover_type
        self.self_adaptive = self_adaptive
        self.mutation_rates = mutation_rates
        self.mutation_step_sizes = mutation_step_sizes
        self.next_idx = 0
        self.population = []
        for i in range(pop_size):
            self.population.append(Individual.random(self.problem, self.next_idx, self.n_poly, self.n_vertex, self.self_adaptive))
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
            pass # Implemented later when selecting individuals for crossover
        else:
            raise ValueError(f'Invalid selection strategy "{self.selection_strategy}"')
        selection_probs = selection_probs / selection_probs.sum()

        # Crossover
        offspring = []
        for i in range(0, self.pop_size):         
            if type(self.selection_strategy) is selection.TournamentSelection:
                tournament = np.random.choice(self.population, size=self.selection_strategy.k*2, replace=False)
                p1, p2 = min(tournament[0::2], key=lambda p: p.fitness), min(tournament[1::2], key=lambda p: p.fitness) # Disjointed tournaments
            else:
                p1, p2 = np.random.choice(self.population, p=selection_probs, size=2, replace=False)
            newind = Individual.crossover(p1, p2, self.crossover_type)
            offspring.append(newind)

        # Mutation
        for ind in offspring:
            self.next_idx = ind.mutate(self.next_idx, self.mutation_rates, self.mutation_step_sizes)

        # Replace old population
        if type(self.replacement_strategy) is replacement.CommaReplacement:
            self.population = offspring
        elif type(self.replacement_strategy) is replacement.PlusReplacement:
            self.population = self.population + offspring
        elif type(self.replacement_strategy) is replacement.CrowdingReplacement:
            for offspring in offspring:
                pool_idx = np.random.choice(len(self.population), size=self.replacement_strategy.pool_size, replace=False)
                closest_idx = min(pool_idx, key=lambda i: Individual.dist(offspring, self.population[i]))
                if offspring.fitness < self.population[closest_idx].fitness: 
                    self.population[closest_idx] = offspring # If offspring fitness is better than closest individual
        else:
            raise ValueError(f'Invalid replacement strategy "{self.replacement_strategy}"')

        # Sort population by fitness
        self.sort_population()
        
        # Survivor selection
        self.population = self.population[:self.pop_size]

        return self.generation, self.population


    def sort_population(self):
        self.population.sort(key=lambda i: i.fitness)


    def diversity(self):
        # Compute pairwise distances
        dist = np.zeros((len(self.population), len(self.population)))
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):
                dist[i, j] = Individual.dist(self.population[i], self.population[j])          
                dist[j, i] = dist[i, j]
        '''
        if self.generation > 0:
            import sklearn.manifold
            import matplotlib.pyplot as plt
            res = sklearn.manifold.TSNE(metric='precomputed', random_state=0).fit_transform(dist)
            plt.scatter(res[:, 0], res[:, 1], s=4, c='r')
            plt.show()
        '''
        return dist.sum()


    def update_target(self, target):
        self.problem.set_target(target)
        