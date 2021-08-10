import numpy as np
from itertools import zip_longest
from numpy.random import randint, rand, normal, choice, shuffle
import cv2 as cv
from PIL import Image, ImageDraw

from ..problem import Problem
from .polygon import Polygon


class Individual:

    ONE_POINT_CROSSOVER = 1
    UNIFORM_CROSSOVER = 2
    ALIGNED_CROSSOVER = 3

    def __init__(self, problem, polygons):
        self.problem = problem
        self.polygons = polygons
        self._fitness = None

    def random(problem, next_idx, n_poly, n_vertex, evolution_strategies):
        # Init random individual
        polygons = [Polygon.random(next_idx + idx, problem, n_vertex, evolution_strategies) for idx in range(n_poly)]
        return Individual(problem, polygons)

    def crossover(parent1, parent2, kind):
        polygons1, polygons2 = [p.copy() for p in parent1.polygons], [p.copy() for p in parent2.polygons]
        if kind == Individual.ONE_POINT_CROSSOVER:
            split_idx = randint(0, max(parent1.n_poly, parent2.n_poly))
            offspring_polygons = polygons1[:split_idx] + polygons2[split_idx:]
        elif kind == Individual.UNIFORM_CROSSOVER:
            offspring_polygons = [polygons1[i] if rand() < 0.5 else polygons2[i] for i in range(min(parent1.n_poly, parent2.n_poly))]  # Common polygons
        elif kind == Individual.ALIGNED_CROSSOVER:
            offspring_polygons = []
            i1, i2 = 0, 0
            while i1 < len(polygons1) and i2 < len(polygons2):
                poly1 = polygons1[i1] if i1 < len(polygons1) else None
                poly2 = polygons2[i2] if i2 < len(polygons2) else None
                # Excess polygons
                if poly1 is None or poly2 is None:
                    if poly1 is not None and parent1.fitness >= parent2.fitness:
                        offspring_polygons.append(poly1)
                    elif poly2 is not None and parent2.fitness >= parent1.fitness:
                        offspring_polygons.append(poly2)
                    i1, i2 = i1+1, i2+1
                # Matching polygons
                elif poly1.idx == poly2.idx:
                    if parent1.fitness >= parent2.fitness:
                        offspring_polygons.append(poly1)
                    elif parent2.fitness >= parent1.fitness:
                        offspring_polygons.append(poly2)
                    i1, i2 = i1+1, i2+1
                # Disjoint polygons
                else:
                    if poly1.idx < poly2.idx:
                        if parent1.fitness >= parent2.fitness or len(offspring_polygons) == 0 or rand() < 0.1:
                            offspring_polygons.append(poly1)
                        i1 = i1+1
                    else:
                        if parent2.fitness >= parent1.fitness or len(offspring_polygons) == 0 or rand() < 0.1:
                            offspring_polygons.append(poly2)
                        i2 = i2+1
        else:
            raise ValueError(f'Invalid crossover kind "{kind}"')
        # Create new individual
        return Individual(parent1.problem, offspring_polygons)

    def mutate(self, next_idx, mutation_chances, mutation_factors):
        # Mutate polygons
        for poly in self.polygons:
            poly.mutate(mutation_chances, mutation_factors)
        # Randomly add a new polygon
        if rand() < mutation_chances[0]:
            self.polygons.append(Polygon.random(next_idx, self.problem, self.polygons[-1].n_vertex, self.polygons[-1].strategy_params is not None))
            next_idx += 1
        # Reset fitness
        self._fitness = None
        # Return true if the current historcial marking has been used
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

    def dist(self, individual):
        dist = 0
        for poly1, poly2 in zip_longest(self.polygons, individual.polygons):
            if poly1 is not None and poly2 is not None:
                dist += poly1.dist(poly2)**2
            else:
                dist += poly1.dist(poly2) if poly1 is not None else poly2.dist(poly1) 
        dist = np.sqrt(dist)
        return dist
        '''
        # OLD, NEAT-based speciation
        excess_count, match_count, disjoint_count = 1, 1, 1
        max_n_poly = max(len(self.polygons), len(individual.polygons)) + 1
        polygons_dist = 0
        i1, i2 = 0, 0
        while i1 < len(self.polygons) and i2 < len(individual.polygons):
            poly1 = self.polygons[i1] if i1 < len(self.polygons) else None
            poly2 = individual.polygons[i2] if i2 < len(individual.polygons) else None
            # Excess polygons
            if poly1 is None or poly2 is None:
                excess_count += 1
                i1, i2 = i1+1, i2+1
            # Matching polygons
            elif poly1.idx == poly2.idx:
                polygons_dist += poly1.dist(poly2)
                match_count += 1
                i1, i2 = i1+1, i2+1
            # Disjoint polygons
            else:
                disjoint_count += 1
                if poly1.idx < poly2.idx:
                    i1 = i1+1
                else:
                    i2 = i2+1
        return (excess_count / max_n_poly) + (disjoint_count / max_n_poly) + (polygons_dist / match_count)  # Inspired by NEAT speciation
        '''

    @property
    def n_poly(self):
        return len(self.polygons)

    @property
    def fitness(self):
        if self._fitness is None:
            if self.problem.problem_type == Problem.RGB:
                self._fitness = np.sum(cv.absdiff(self.draw(full_res=False), self.problem.target).astype(np.int)**2)
            else:
                image = cv.cvtColor(self.draw(full_res=False), cv.COLOR_BGR2GRAY)
                self._fitness = np.sum(self.problem.target[image > 0].astype(np.int)**2)
        return self._fitness
