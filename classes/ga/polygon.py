import numpy as np
from numpy import random
from numpy.random import randint, rand
from numba import njit

from ..utils import clip, normal

PTS_RADIUS = 0.3  # Maximm distance radius of generated points in first initialization.
LINE_LENGTH = 20
ALPHA_MIN, ALPHA_MAX = 40, 200


class Polygon:
    def __init__(self, img_size, pts, color, alpha):
        self.img_size = img_size
        self.pts = pts
        self.color = color
        self.alpha = alpha

    def random(problem, n_vertex):
        # Initialize the polygon's points randomly
        img_size = np.array(problem.target.shape[:2][::-1])
        pos = randint(low=0, high=img_size, size=2)  # Generale position of the polygon, to which start creating points
        if n_vertex == 2:
            # Create a line with random angle and center on pos
            theta = rand() * 2 * np.pi
            d = [LINE_LENGTH * np.cos(theta), LINE_LENGTH * np.sin(theta)]
            pts = np.array([pos-d, pos+d])
        else:
            # Random points inside the radius defined
            radius = (img_size * PTS_RADIUS).astype(np.int)
            pts = randint(low=pos-radius, high=pos+radius, size=(n_vertex, 2))  # Create
            pts = np.clip(pts, [0, 0], img_size)  # Clip points outside limits
        color = randint(0, 256, (3))  # RGB
        alpha = randint(ALPHA_MIN, ALPHA_MAX)  # Alpha channel
        return Polygon(img_size, pts, color, alpha)

    def mutate(self, pts_chance, color_chance, alpha_chance, pts_factor, color_factor, alpha_factor):
        self.pts, self.color, self.alpha = Polygon._mutate(self.img_size, self.pts, self.color, self.alpha ,pts_chance, color_chance, alpha_chance, pts_factor, color_factor, alpha_factor)

    @njit
    def _mutate(img_size, pts, color, alpha, pts_chance, color_chance, alpha_chance, pts_factor, color_factor, alpha_factor):
        pts_factor = img_size.max() * pts_factor
        color_factor = 255 * color_factor
        alpha_factor = (ALPHA_MAX - ALPHA_MIN) * alpha_factor
        if pts.shape[0] == 2:
            # Recompute center point and rotation angle
            center = np.array([pts[:, i].mean() for i in range(pts.shape[1])])
            theta = np.arctan((center[1] - pts[0, 1]) / (center[0] - pts[0, 0]))
            # Mutate center and rotation
            if rand() < pts_chance:
                center = center + normal(scale=pts_factor//4)
                center[0], center[1] = clip(center[0], 0, img_size[0]), clip(center[1], 0, img_size[1])
            if rand() < pts_chance:
                theta = theta + normal(scale=np.pi/4)
            # Compute new point coordinates
            d = np.array([LINE_LENGTH * np.cos(theta), LINE_LENGTH * np.sin(theta)])
            pts[0], pts[1] = center-d, center+d
        else:
            # Mutate points
            for i, pt in enumerate(pts):
                for j, x in enumerate(pt):
                    if rand() < pts_chance:
                        pts[i, j] = int(clip(x + normal(scale=pts_factor//4), 0, img_size[j]))
                        #pts[i, j] = int(x + randint(-pts_factor//2, pts_factor//2))
            # Mutate color
            for i, c in enumerate(color):
                if rand() < color_chance:
                    color[i] = int(clip(c + normal(scale=color_factor//4), 0, 255))
                    #color[i] = int(clip(c + randint(-color_factor//2, color_factor//2)))
            # Mutate alpa
            if rand() < alpha_chance:
                alpha = int(clip(alpha + normal(scale=alpha_factor//4), ALPHA_MIN, ALPHA_MAX))
                #alpha = int(clip(alpha + randint(-alpha_factor//2, alpha_factor//2)))
        return pts, color, alpha

    @property
    def n_vertex(self):
        return len(self.pts)

    def copy(self):
        return Polygon(self.img_size.copy(), self.pts.copy(), self.color.copy(), self.alpha)
