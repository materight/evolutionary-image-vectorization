import numpy as np
import cv2 as cv

class Individual:

    def __init__(self, target, n_poly, n_vertex, genes=None): 
        self.target = target
        self.n_poly = n_poly
        self.n_vertex = n_vertex

        # Init random individual
        self._fitness = None
        if genes is None:
            self.genes = np.random.rand(n_poly * (n_vertex * 2 + 3 + 1))
        else:
            self.genes = genes


    def crossover(parent1, parent2):
        offspring = np.zeros_like(parent1.genes)
        split_idx = np.random.randint(0, parent1.n_poly) * (parent1.n_vertex * 2 + 3 + 1)
        for i, (g1, g2) in enumerate(zip(parent1.genes, parent2.genes)):
            offspring[i] = g1 if i < split_idx else g2
        return Individual(parent1.target, parent1.n_poly, parent1.n_vertex, offspring)
        

    def to_data(self):
        polys = np.split(self.genes, self.n_poly)
        polys = [np.split(poly, [-4, -1]) for poly in polys]
        for i, poly in enumerate(polys):
            polys[i][0] = (poly[0].reshape((self.n_vertex, 2)) * np.array([(self.target.shape[1] - 1), (self.target.shape[0] - 1)])).astype(int) # pts
            polys[i][1] = (poly[1] * 255).astype(int) # color
            polys[i][2] = (poly[2][0] * .05 + .3) # alpha between 0.05 and 0.3
        return polys

    def draw(self):
        canvas = np.zeros_like(self.target)
        for pts, color, alpha in self.to_data():
            mask = cv.fillPoly(np.zeros_like(canvas), [pts], (1, 1, 1))
            canvas = np.where(mask > 0, color * alpha + canvas * (1 - alpha), canvas).astype(np.uint8)
        return canvas

    def fitness(self):
        if self._fitness is None:
            canvas = self.draw()
            self._fitness = np.sum((canvas - self.target)**2)
        return self._fitness

    def mutate(self, mutation_chance=0.1):
        for i, gene in enumerate(self.genes):
            if np.random.rand() < mutation_chance:
                self.genes[i] = max(0, min(1, self.genes[i] + np.random.normal(scale=0.1)))
        self._fitness = None

