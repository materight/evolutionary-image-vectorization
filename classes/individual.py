from classes.polygon import Polygon
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw

from .polygon import Polygon



class Individual:

    def __init__(self, target, n_poly, n_vertex, polygons=None): 
        self.target = target
        self.n_poly = n_poly
        self.n_vertex = n_vertex

        # Init random individual
        self._fitness = None
        if polygons is None:
            self.polygons = [Polygon(target, n_vertex) for i in range(n_poly)]
        else:
            self.polygons = polygons


    def crossover(parent1, parent2):
        split_idx = np.random.randint(0, parent1.n_poly) * (parent1.n_vertex * 2 + 3 + 1)
        offspring_polygons = [poly1 if i < split_idx else poly2 for i, (poly1, poly2) in enumerate(zip(parent1.polygons, parent2.polygons))]
        return Individual(parent1.target, parent1.n_poly, parent1.n_vertex, offspring_polygons)
        

    def draw(self):
        canvas = Image.new('RGB', (self.target.shape[1], self.target.shape[0]), color='black')
        draw = ImageDraw.Draw(canvas, 'RGBA')
        for poly in self.polygons:
            draw.polygon([tuple(p) for p in poly.pts], fill=(*poly.color, poly.alpha))
        canvas = cv.cvtColor(np.array(canvas), cv.COLOR_RGB2BGR)
        return canvas


    def fitness(self):
        if self._fitness is None:
            canvas = self.draw()
            self._fitness = np.sum((canvas - self.target)**2)
        return self._fitness


    def mutate(self, mutation_chance=0.01, mutation_factors=(5, 5, 1)):
        for poly in self.polygons:
            poly.mutate(mutation_chance, *mutation_factors)
        self._fitness = None

