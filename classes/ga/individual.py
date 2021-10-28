import numpy as np
from itertools import zip_longest
from numpy.lib.function_base import average
from numpy.random import randint, rand, normal, choice, shuffle
import cv2 as cv
from PIL import Image, ImageDraw

from ..problem import Problem
from ..operators import crossover
from .polygon import Polygon


class Individual:

    def __init__(self, problem, polygons):
        self.problem = problem
        self.polygons = polygons
        self._fitness = None


    def random(problem, next_idx, n_poly, n_vertex, radom_color, self_adaptive):
        # Init random individual
        polygons = [Polygon.random(next_idx + idx, problem, n_vertex, radom_color, self_adaptive) for idx in range(n_poly)]
        return Individual(problem, polygons)


    def crossover(parent1, parent2, kind):
        polygons1, polygons2 = [p.copy() for p in parent1.polygons], [p.copy() for p in parent2.polygons]
        if isinstance(kind, crossover.OnePointCrossover):
            split_idx = randint(0, max(parent1.n_poly, parent2.n_poly))
            offspring_polygons = polygons1[:split_idx] + polygons2[split_idx:]
        elif isinstance(kind, crossover.UniformCrossover):
            offspring_polygons = [polygons1[i] if rand() < 0.5 else polygons2[i] for i in range(min(parent1.n_poly, parent2.n_poly))]  # Common polygons
        elif isinstance(kind, crossover.ArithmeticCrossover):
            offspring_polygons = [Polygon.average(polygons1[i], polygons2[i]) for i in range(min(parent1.n_poly, parent2.n_poly))]
        else:
            raise ValueError(f'Invalid crossover kind "{kind}"')
        # Create new individual
        return Individual(parent1.problem, offspring_polygons)


    def mutate(self, next_idx, mutation_chances, mutation_factors):
        # Mutate polygons
        for poly in self.polygons:
            poly.mutate(mutation_chances, mutation_factors)
        # Randomly add a new polygon
        '''
        if rand() < mutation_chances[0]:
            self.polygons.append(Polygon.random(next_idx, self.problem, self.polygons[-1].n_vertex, self.polygons[-1].strategy_params is not None))
            next_idx += 1
        '''
        # Reset fitness of mutated individual
        self._fitness = None
        # Return the next historical marking value
        return next_idx


    def draw(self, full_res=True):
        scale = 1/self.problem.scale_factor if full_res else 1  # Rescale internal image target to full scale
        img = Image.new('RGB', (int(self.problem.target.shape[1]*scale), int(self.problem.target.shape[0]*scale)), color='black')
        draw = ImageDraw.Draw(img, 'RGBA')
        for poly in self.polygons:
            if poly.pts.shape[0] == 2:
                draw.line([tuple(p) for p in np.floor(poly.pts*scale).astype(np.int)], fill=(255, 255, 255), width=int(scale))
            else:
                draw.polygon([tuple(p) for p in np.floor(poly.pts*scale).astype(np.int)], fill=(*poly.color.astype(np.int), int(poly.alpha)))
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        return img


    def dist(ind1, ind2):
        dist = 0
        for poly1, poly2 in zip_longest(ind1.polygons, ind2.polygons):
            dist += poly1.dist(poly2)**2
        return np.sqrt(dist)


    @property
    def n_poly(self):
        return len(self.polygons)


    @property
    def fitness(self):
        if self._fitness is None:
            self._fitness = np.sum(cv.absdiff(self.draw(full_res=False), self.problem.target).astype(np.int)**2) / self.problem.target.size
        return self._fitness


    @property
    def fitness_perc(self):
        return 1 - self.fitness / 256**2 # Fitness value over maximum possible fitness


    def copy(self):
        return Individual(self.problem, [p.copy() for p in self.polygons])        
