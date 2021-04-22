import numpy as np
from numpy.random import randint, rand, normal


PTS_RADIUS = 100 # Maximm distance radius of generatedpojints in first initialization
ALPHA_MIN, ALPHA_MAX = 20, 100



class Polygon:
    def __init__(self, target, n_vertex):
        self.n_vertex = n_vertex
        self.img_size = target.shape[:2][::-1]
        # Initialize the polygon's points randomly
        pos = randint(0, self.img_size, 2) # Generale position of the polygon, to which start creating points
        pts = randint(pos-PTS_RADIUS, pos+PTS_RADIUS, (n_vertex, 2)) # Create 
        pts = np.clip(pts, [0, 0], self.img_size) # Clip points outside limits
        self.pts = pts
        self.color = randint(0, 256, (3)) # RGB
        self.alpha = randint(ALPHA_MIN, ALPHA_MAX) # Alpha channel


    def mutate(self, chance, pts_factor, color_factor, alpha_factor):
        # Mutate points
        mutation = np.clip(self.pts + normal(scale=pts_factor, size=self.pts.shape), [0, 0], self.img_size).astype(int)
        mask = rand(*self.pts.shape) < chance
        self.pts[mask] = mutation[mask]
        # Mutate colors
        mutation = np.clip(self.color + normal(scale=color_factor, size=self.color.shape), 0, 255).astype(int)
        mask = rand(*self.color.shape) < chance
        self.color[mask] = mutation[mask]
        # Mutate alpha
        mutation = np.clip(int(self.alpha + normal(scale=alpha_factor)), ALPHA_MIN, ALPHA_MAX)
        if rand() < chance: 
            self.alpha = mutation