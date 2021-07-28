import numpy as np
from numpy import random
from numpy.random import randint, rand
from numba import njit

from ..utils import clip, normal

PTS_RADIUS = 0.3  # Maximm distance radius of generated points in first initialization.
LINE_LENGTH = 10
ALPHA_MIN, ALPHA_MAX = 10, 250


class Polygon:
    def __init__(self, idx, img_size, pts, color, alpha):
        self.idx = idx  # Historical marking 
        self.img_size = img_size
        self.pts = pts
        self.color = color
        self.alpha = alpha


    def random(idx, problem, n_vertex):
        # Initialize the polygon's points randomly
        img_size = np.array(problem.target.shape[:2][::-1])
        pos = randint(low=0, high=img_size, size=2)  # Generale position of the polygon, to which start creating points        
        # Random points inside the radius defined
        radius = (img_size * PTS_RADIUS).astype(np.int)
        pts = randint(low=pos-radius, high=pos+radius, size=(n_vertex, 2))  # Create
        pts = np.clip(pts, [0, 0], img_size)  # Clip points outside limits
        color = randint(0, 256, (3))  # RGB
        alpha = randint(ALPHA_MIN, ALPHA_MAX)  # Alpha channel
        return Polygon(idx, img_size, pts, color, alpha)


    def mutate(self, pts_chance, color_chance, alpha_chance, pts_factor, color_factor, alpha_factor):
        self.pts, self.color, self.alpha = Polygon._mutate(self.img_size, self.pts, self.color, self.alpha ,pts_chance, color_chance, alpha_chance, pts_factor, color_factor, alpha_factor)


    @njit
    def _mutate(img_size, pts, color, alpha, pts_chance, color_chance, alpha_chance, pts_factor, color_factor, alpha_factor):
        pts_factor = img_size.max() * pts_factor
        color_factor = 255 * color_factor
        alpha_factor = (ALPHA_MAX - ALPHA_MIN) * alpha_factor
        # Mutate points
        for i, pt in enumerate(pts):
            for j, x in enumerate(pt):
                if rand() < pts_chance:
                    pts[i, j] = int(clip(x + normal(scale=pts_factor//4), 0, img_size[j]))
        # Mutate color
        for i, c in enumerate(color):
            if rand() < color_chance:
                color[i] = int(clip(c + normal(scale=color_factor//4), 0, 255))
        # Mutate alpa
        if rand() < alpha_chance:
            alpha = int(clip(alpha + normal(scale=alpha_factor//4), ALPHA_MIN, ALPHA_MAX))
        return pts, color, alpha


    def dist(self, poly):
        return Polygon._dist(self.pts, self.color, self.alpha, poly.pts, poly.color, poly.alpha, self.img_size)


    @njit
    def _dist(poly1_pts, poly1_color, poly1_alpha, poly2_pts, poly2_color, poly2_alpha, img_size):
        pts_dist = (np.abs(poly1_pts - poly2_pts) / img_size).mean()
        color_dist = (np.abs(poly1_color - poly2_color) / 256).mean()
        alpha_dist = (np.abs(poly1_alpha - poly2_alpha) / (ALPHA_MAX - ALPHA_MIN))
        return (pts_dist + color_dist + alpha_dist) / 3


    @property
    def n_vertex(self):
        return len(self.pts)


    def copy(self):
        return Polygon(self.idx, self.img_size.copy(), self.pts.copy(), self.color.copy(), self.alpha)
