import numpy as np

from ..problem import Problem
from .individual import Individual
from ..operators import selection, replacement

class GA:
    
    def __init__(self, target, pop_size, n_poly, n_vertex, selection_strategy, replacement_strategy, crossover_type, self_adaptive, mutation_rates, mutation_step_sizes, internal_resolution=75):
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
        if isinstance(self.selection_strategy, selection.RouletteWheelSelection): # Roulette wheel selection (fitness-proportionate)
            selection_probs = np.array([i.fitness for i in self.population])
        elif isinstance(self.selection_strategy, selection.RankBasedSelection): # Rank-based selection
            selection_probs = np.arange(len(self.population), 0, -1)
        elif isinstance(self.selection_strategy, selection.TruncatedSelection): # Truncated rank-based selection
            selection_count = max(int(len(self.population) * self.selection_strategy.selection_cutoff), 2)
            selection_probs = np.array([1 if i < selection_count else 0 for i in range(len(self.population))])
        elif isinstance(self.selection_strategy, selection.TournamentSelection): # Tournament selection
            pass # Implemented later when selecting individuals for crossover
        else:
            raise ValueError(f'Invalid selection strategy "{self.selection_strategy}"')
        selection_probs = selection_probs / selection_probs.sum()

        # Crossover
        offspring = []
        for i in range(0, self.pop_size):         
            if isinstance(self.selection_strategy, selection.TournamentSelection):
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
        if isinstance(self.replacement_strategy, replacement.CommaReplacement):
            self.population = offspring
        elif isinstance(self.replacement_strategy, replacement.PlusReplacement):
            self.population = self.population + offspring
        elif isinstance(self.replacement_strategy, replacement.CrowdingReplacement):
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
        return dist


    def update_target(self, target):
        self.problem.set_target(target)
        