import numpy as np
from numpy import random
from numpy.random import randint, rand, normal


PTS_RADIUS = 0.5  # Maximm distance radius of generated points in first initialization.
ALPHA_MIN, ALPHA_MAX = 60, 180


class Polygon:
    def __init__(self, img_size, pts, color, alpha):
        self.img_size = img_size
        self.pts = pts
        self.color = color
        self.alpha = alpha

    def random(target, n_vertex):
        # Initialize the polygon's points randomly
        img_size = np.array(target.shape[:2][::-1])
        pos = randint(low=0, high=img_size, size=2)  # Generale position of the polygon, to which start creating points
        radius = (img_size * PTS_RADIUS).astype(np.int)
        pts = randint(low=pos-radius, high=pos+radius, size=(n_vertex, 2))  # Create
        pts = np.clip(pts, [0, 0], img_size)  # Clip points outside limits
        color = randint(0, 256, (3))  # RGB
        alpha = randint(ALPHA_MIN, ALPHA_MAX)  # Alpha channel
        return Polygon(img_size, pts, color, alpha)

    def mutate(self, chance, pts_factor, color_factor, alpha_factor):
        pts_factor = self.img_size.max() * pts_factor
        color_factor = 255 * color_factor
        alpha_factor = (ALPHA_MAX - ALPHA_MIN) * alpha_factor
        # Mutate points
        for i, pt in enumerate(self.pts):
            for j, x in enumerate(pt):
                if rand() < chance:
                    #self.pts[i, j] = int(np.clip(x + normal(scale=pts_factor), 0, self.img_size[j]))
                    self.pts[i, j] = int(np.clip(x + randint(-pts_factor//2, pts_factor//2), 0, self.img_size[j]))
        # Mutate color
        for i, c in enumerate(self.color):
            if rand() < chance:
                #self.color[i] = int(np.clip(c + normal(scale=color_factor), 0, 255))
                self.color[i] = int(np.clip(c + randint(-color_factor//2, color_factor//2), 0, 255))
        # Mutate alpa
        if rand() < chance:
            #self.alpha = int(np.clip(self.alpha + normal(scale=alpha_factor), ALPHA_MIN, ALPHA_MAX))
            self.alpha = int(np.clip(self.alpha + randint(-alpha_factor//2, alpha_factor//2), ALPHA_MIN, ALPHA_MAX))

    @property
    def n_vertex(self):
        return len(self.pts)

    def copy(self):
        return Polygon(self.img_size.copy(), self.pts.copy(), self.color.copy(), self.alpha)
