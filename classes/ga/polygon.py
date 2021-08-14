import numpy as np
from numpy import random
from numpy.random import randint, rand
from numba import njit

from ..utils import clip, normal, uniform

PTS_RADIUS = 0.3  # Maximm distance radius of generated points in first initialization.
ALPHA_MIN, ALPHA_MAX = 0, 256


class Polygon:
    def __init__(self, idx, img_size, pts, color, alpha, strategy_params):
        self.idx = idx  # Historical marking
        self.img_size = img_size
        self.pts = pts
        self.color = color
        self.alpha = alpha
        self.strategy_params = strategy_params

    def random(idx, problem, n_vertex, self_adaptive):
        # Initialize the polygon's points randomly
        img_size = np.array(problem.target.shape[:2][::-1])
        pos = rand(2) * img_size  # Generale position of the polygon, to which start creating points
        # Random points inside the radius defined
        radius = (img_size * PTS_RADIUS).astype(np.int)
        pts = rand(n_vertex, 2)*2*radius + (pos - radius) # Create
        pts = np.clip(pts, [0, 0], img_size)  # Clip points outside limits
        color = rand(3) * 256  # RGB
        alpha = rand() * (ALPHA_MAX - ALPHA_MIN) + ALPHA_MIN  # Alpha channel
        # Init ES parameters
        strategy_params = rand(3)/100 if self_adaptive else None
        return Polygon(idx, img_size, pts, color, alpha, strategy_params)

    def mutate(self, mutation_chances, mutation_factors):
        self.pts, self.color, self.alpha, self.strategy_params = Polygon._mutate(self.img_size, self.pts, self.color, self.alpha, mutation_chances, mutation_factors, self.strategy_params)
        
    @njit
    def _mutate(img_size, pts, color, alpha, mutation_chances, mutation_factors, strategy_params):
        # Genetic Algorithm
        if strategy_params is None:
            pts_chance, color_chance, alpha_chance = mutation_chances
            pts_factor = (img_size.max() * mutation_factors[0]) / 4
            color_factor = (255 * mutation_factors[1]) / 4
            alpha_factor = ((ALPHA_MAX - ALPHA_MIN) * mutation_factors[2]) / 4
        # Evolution Strategies
        else:
            pts_chance, color_chance, alpha_chance = (1, 1, 1)
            n = pts.size + color.size + 1 # Total size of genotype 
            tau, tau1 = 1/np.sqrt(2*np.sqrt(n)), 1/np.sqrt(2*n) 
            epsilon = 0.00001
            for i, sigma in enumerate(strategy_params):
                strategy_params[i] = sigma * np.exp(tau * normal() + tau1 * normal())
                if strategy_params[i] < epsilon: strategy_params[i] = epsilon
            pts_factor = img_size.max() * strategy_params[0]
            color_factor = 255  * strategy_params[1]
            alpha_factor = (ALPHA_MAX - ALPHA_MIN) * strategy_params[2]
            #print('ES:', int(pts_factor), int(color_factor), int(alpha_factor))
            print('ES:', strategy_params[0], strategy_params[1], strategy_params[2])
        # Mutate points
        for i, pt in enumerate(pts):
            for j, x in enumerate(pt):
                if rand() < pts_chance:
                    pts[i, j] = clip(x + normal(scale=pts_factor), 0, img_size[j])
        # Mutate color
        for i, c in enumerate(color):
            if rand() < color_chance:
                color[i] = clip(c + normal(scale=color_factor), 0, 255)
        # Mutate alpa
        if rand() < alpha_chance:
            alpha = clip(alpha + normal(scale=alpha_factor), ALPHA_MIN, ALPHA_MAX)
        return pts, color, alpha, strategy_params


    def average(poly1, poly2):
        avg_pts = (poly1.pts + poly2.pts) / 2
        avg_color = (poly1.color + poly2.color) / 2
        avg_alpha = (poly1.alpha + poly2.alpha) / 2
        avg_strategy_params = (poly1.strategy_params + poly2.strategy_params) / 2 if poly1.strategy_params is not None else None
        return Polygon(poly1.idx, poly1.img_size.copy(), avg_pts, avg_color, avg_alpha, avg_strategy_params)


    def dist(self, poly):
        if poly is None: # Maximum possible distance
            return len(self.pts) + len(self.color) + 1
        else:
            return Polygon._dist(self.pts, self.color, self.alpha, poly.pts, poly.color, poly.alpha, self.img_size)

    @njit
    def _dist(poly1_pts, poly1_color, poly1_alpha, poly2_pts, poly2_color, poly2_alpha, img_size):
        pts_dist = ((poly1_pts - poly2_pts) / img_size)**2
        color_dist = ((poly1_color - poly2_color) / 256)**2
        alpha_dist = ((poly1_alpha - poly2_alpha) / (ALPHA_MAX - ALPHA_MIN))**2
        return np.sqrt(np.sum(pts_dist) + np.sum(color_dist) + np.sum(alpha_dist)) # Normalized euclidean distance

    @property
    def n_vertex(self):
        return len(self.pts)

    @property
    def area(self):
        return Polygon._area(self.pts)

    @njit
    def _area(pts):
        # Calculate polygon area using Shoelace formula
        area = 0
        n_vertex = len(pts)
        j = n_vertex - 1
        for i in range(0, n_vertex):
            area += (pts[j, 0] + pts[i, 0]) * (pts[j, 1] - pts[i, 1])
            j = i
        return int(abs(area / 2.0))  # Return absolute value

    def copy(self):
        return Polygon(self.idx, self.img_size.copy(), self.pts.copy(), self.color.copy(), self.alpha, self.strategy_params.copy() if self.strategy_params is not None else None)
