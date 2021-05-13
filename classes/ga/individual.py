import numpy as np
from numpy.random import randint, rand, normal, choice, shuffle
import cv2 as cv
from PIL import Image, ImageDraw

from ..problem import Problem
from .polygon import Polygon


class Individual:

    def __init__(self, problem, polygons):
        self.problem = problem
        self.polygons = polygons
        self._fitness = None

    def random(problem, n_poly, n_vertex):
        # Init random individual
        polygons = [Polygon.random(problem, n_vertex) for i in range(n_poly)]
        return Individual(problem, polygons)

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
        #offspring_polygons = [polygons1[i] if rand() < 0.5 else polygons2[i] for i in range(min(parent1.n_poly, parent2.n_poly))]
        offspring_polygons = []
        THRESH = 0.5 # parent1.fitness / (parent1.fitness + parent2.fitness) # Proportional to fitness
        for i in range(min(parent1.n_poly, parent2.n_poly)):
            pts, color, alpha = np.zeros_like(polygons1[i].pts), np.zeros_like(polygons1[i].color), None
            for j in range(pts.shape[0]):
                for k in range(pts.shape[1]):
                    pts[j, k] = polygons1[i].pts[j, k] if rand() < THRESH else polygons2[i].pts[j, k]
            for j in range(color.shape[0]):
                color[j] = polygons1[i].color[j] if rand() < THRESH else polygons2[i].color[j]
            alpha = polygons1[i].alpha if rand() < THRESH else polygons2[i].alpha
            offspring_polygons.append(Polygon(polygons1[i].img_size.copy(), pts, color, alpha))

        # Create new individual
        return Individual(parent1.problem, offspring_polygons)

    def mutate(self, mutation_chances, mutation_factors):
        # Muatate polygons
        for poly in self.polygons:
            poly.mutate(*mutation_chances, *mutation_factors)
        # Randomly replace two polygons' positions
        if rand() < mutation_chances[0]:
            i1, i2 = randint(0, self.n_poly, 2)
            self.polygons[i1], self.polygons[i2] = self.polygons[i2], self.polygons[i1]
        # Reset fitness and image value
        self._fitness = None

    def draw(self, full_res=True):
        scale = 1/self.problem.scale_factor if full_res else 1  # Rescale internal image target to full scale
        img = Image.new('RGB', (int(self.problem.target.shape[1]*scale), int(self.problem.target.shape[0]*scale)), color='black')
        draw = ImageDraw.Draw(img, 'RGBA')
        for poly in self.polygons:
            if poly.pts.shape[0] == 2:
                draw.line([tuple(p) for p in np.floor(poly.pts*scale)], fill=(255, 255, 255), width=int(scale))
            else:
                draw.polygon([tuple(p) for p in np.floor(poly.pts*scale)], fill=(*poly.color, poly.alpha))
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        return img

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
