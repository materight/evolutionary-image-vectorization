import numpy as np
import math
from numpy.random import randint, rand, normal, choice, shuffle


class Line:
    def __init__(self, x):
        self.x = x

    def random(problem):
        img_size = np.array(problem.target.shape[:2][::-1])
        center = randint(low=0, high=img_size, size=2)
        rotation = rand()*2*math.pi # Rotation radians
        length = 20 #randint(LENGTH_MIN, LENGTH_MAX)
        return Line(np.array([*center, rotation, length], dtype=np.float64))

    def diff(self, line):
        return self.x - line.x

    def update(self, velocity):
        return Line(self.x.copy() + velocity)

    @property
    def center(self):
        return self.x[:2]
    
    @property
    def rotation(self):
        return self.x[2]

    @property
    def length(self):
        return self.x[3]

    @property
    def coords(self):
        r = self.length / 2
        d = [r * math.cos(self.rotation), r * math.sin(self.rotation)] # Compute displacement of line ends from line center
        return np.concatenate([self.center + d, self.center - d]) 
