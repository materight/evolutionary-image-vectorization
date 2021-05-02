
import numpy as np
import cv2 as cv
from numpy.random import randint, shuffle, choice


class PSO():
    def __init__(self, target, pop_size=50, internal_res=150):
        self.generation = 0
        self.scale_factor = internal_res / min(target.shape[:2])
        self.target = cv.resize(target, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        print('Internal img size: ', self.target.shape)
        
        # Compute fitness landscape
        gray_filtered = cv.bilateralFilter(self.target, 7, 50, 50)
        img = cv.Canny(gray_filtered, 120, 140)
        img = np.where(img > 0, 0, 255).astype(np.uint8)
        field = cv.distanceTransform(img.copy(), cv.DIST_L2, 3)
        field = cv.normalize(field, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        self.target = field

        # Init population
        self.population = []
        for i in range(pop_size):
            pass
            #self.population.append(Individual.random(self.target, self.scale_factor, self.n_poly, self.n_vertex))
        self.population.sort(key=lambda i: i.fitness)

