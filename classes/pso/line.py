import numpy as np
from numpy.random import randint, rand, normal, choice, shuffle
from ..utils import compute_line_coords

class Line:
    def __init__(self, img_size, length, x):
        self.img_size = img_size
        self.length = length
        self.x = x

    def random(problem, length):
        img_size = np.array(problem.target.shape[:2][::-1])
        center = randint(low=length//2, high=img_size-length//2, size=2)
        theta = rand()*2*np.pi # Rotation angle in radians
        return Line(img_size, length, np.array([*center, theta], dtype=np.float64))

    def diff(self, line):
        return self.x - line.x

    def dist(self, line):
        return np.sum((self.center - line.center)**2)**.5

    def update(self, velocity):
        self.x += velocity
        # Clip position if outside image borders
        self.x[:2] = np.clip(self.x[:2], self.length//2 + 1, self.img_size - self.length//2 - 1)
        # Keep rotation angle in [0, 2*pi] to avoid numerical instability
        self.x[2] =  self.x[2] % (np.pi * 2)

    def copy(self):
        return Line(self.img_size.copy(), self.length, self.x.copy())

    @property
    def center(self):
        return self.x[:2]

    @property
    def rotation(self):
        return self.x[2]

    @property
    def filter_coords(self):
        r = self.length / 2
        d = [r * np.cos(self.rotation), r * np.sin(self.rotation)] # Compute displacement of line ends from line center
        cd = [np.cos(self.rotation+np.pi/2), np.sin(self.rotation+np.pi/2)] # Compute displacement of filters centers from line center (perpendicular to line)
        centerL, centerR = self.center + cd, self.center - cd
        coords = np.concatenate([self.center + d, self.center - d])
        coordsL = np.concatenate([centerL + d, centerL - d])
        coordsR = np.concatenate([centerR + d, centerR - d])
        return coords, coordsL, coordsR

    @property
    def coords(self): # Coordinates of drawn line
        return compute_line_coords(self.center, self.rotation, self.length) 

    @property
    def coordsL(self): # Coordinates of left filter
        return compute_line_coords(self.center, self.rotation, self.length, -1) 

    @property
    def coordsR(self): # Coordinates of right filter
        return compute_line_coords(self.center, self.rotation, self.length, +1) 

    @property
    def size(self):
        return self.x.size