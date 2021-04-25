from classes.polygon import Polygon
import numpy as np
from numpy.random import randint, rand, normal, choice
import cv2 as cv
from PIL import Image, ImageDraw
from .polygon import Polygon


class Individual:

    def __init__(self, target, polygons):
        self.target = target
        self.polygons = polygons
        self._img, self._fitness = None, None


    def random(target, n_poly, n_vertex): 
        # Init random individual
        polygons = [Polygon.random(target, n_vertex) for i in range(n_poly)]
        return Individual(target, polygons)


    def crossover(parent1, parent2):
        polygons1, polygons2 = [p.copy() for p in parent1.polygons], [p.copy() for p in parent2.polygons]
        '''
        # One parent
        offspring_polygons = polygons1 if rand() < 0.5 else polygons2
        '''
        '''
        # One-point split
        split_idx = np.random.randint(0, max(parent1.n_poly, parent2.n_poly))
        offspring_polygons = polygons1[:split_idx] + polygons2[split_idx:]
        '''
        # Uniform
        offspring_polygons = [polygons1[i] if rand() < 0.5 else polygons2[i] for i in range(min(parent1.n_poly, parent2.n_poly))]
        # Create new individual
        return Individual(parent1.target, offspring_polygons)

    def mutate(self, mutation_chance=0.05, mutation_factors=(0.2, 0.2, 0.2)):
        # Muatate polygons
        for poly in self.polygons:
            poly.mutate(mutation_chance, *mutation_factors)
        '''
        # Randomly add a polygon. Maximum of 200 polygons.
        if self.n_poly < 200 and rand() < mutation_chance:
            self.polygons.insert(randint(0, self.n_poly), Polygon.random(self.target, self.polygons[0].n_vertex))
        # Randomly remove a polygon. Minimum of 20 polygons
        if self.n_poly > 20 and rand() < mutation_chance:
            self.polygons.pop(randint(0, self.n_poly))
        # Randomly replace two polygons' positions
        if rand() < mutation_chance:
            i1, i2 = randint(0, self.n_poly, 2)
            self.polygons[i1], self.polygons[i2] = self.polygons[i2], self.polygons[i1]
        '''
        # Reset fitness and image value
        self._img, self._fitness = None, None


    @property
    def n_poly(self):
        return len(self.polygons)


    @property
    def img(self):
        if self._img is None:
            self._img = Image.new('RGB', (self.target.shape[1], self.target.shape[0]), color='black')
            draw = ImageDraw.Draw(self._img, 'RGBA')
            for poly in self.polygons:
                if len(poly.pts) == 2:
                    draw.line([tuple(p) for p in poly.pts], width=1)
                else:
                    draw.polygon([tuple(p) for p in poly.pts], fill=(*poly.color, poly.alpha))
            self._img = np.array(self._img)
            self._img = cv.cvtColor(self._img, cv.COLOR_RGB2BGR)
        return self._img


    @property
    def fitness(self):
        if self._fitness is None:
            self._fitness = np.sum((cv.absdiff(self.img, self.target))**2)
        return self._fitness