import numpy as np
from numpy.random import randint, rand, normal, choice, shuffle

LENGTH = 1

class Line:
    def __init__(self, img_size, x):
        self.img_size = img_size
        self.x = x

    def random(problem):
        img_size = np.array(problem.target.shape[:2][::-1])
        center = randint(low=LENGTH//2, high=img_size-LENGTH//2, size=2)
        theta = rand()*2*np.pi # Rotation angle in radians
        return Line(img_size, np.array([*center, theta], dtype=np.float64))

    def diff(self, line):
        return self.x - line.x

    def dist(self, line):
        return np.sum((self.center - line.center)**2)**.5

    def update(self, velocity):
        self.x += velocity
        # Clip position if outside image borders
        self.x[:2] = np.clip(self.x[:2], LENGTH//2 + 1, self.img_size - LENGTH//2 - 1) 

    def copy(self):
        return Line(self.img_size.copy(), self.x.copy())

    @property
    def center(self):
        return self.x[:2]

    @property
    def rotation(self):
        return self.x[2]

    @property
    def coords(self):
        r = LENGTH / 2
        d = [r * np.cos(self.rotation), r * np.sin(self.rotation)] # Compute displacement of line ends from line center
        return np.concatenate([self.center + d, self.center - d]) 

    @property
    def size(self):
        return self.x.size