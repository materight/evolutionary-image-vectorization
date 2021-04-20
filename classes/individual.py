import numpy as np
import cv2 as cv

class Individual:

    def __init__(self, img_shape, n_poly, n_vertex):
        # Init random individual 
        self.img_shape = img_shape
        self.polys = np.random.randint((0, 0), img_shape[1::-1], (n_poly, n_vertex, 2))
        self.colors = np.random.randint(0, 256, (n_poly, 4)) # R, G, B
        self.alphas = np.full((n_poly), .2) # np.random.randint(0, .5, (n_poly))

    def _draw(self):
        result = np.zeros(self.img_shape, np.uint8)
        for pts, color, alpha in zip(self.polys, self.colors, self.alphas):
            layer = np.zeros_like(result)
            cv.fillPoly(layer, [pts], color.tolist())
            result = cv.addWeighted(result, 1, layer, alpha, 0)
        return result

    def fitness(self, target):
        return np.sum((self.canvas - target)**2) 

    def mutate(self):
        pass