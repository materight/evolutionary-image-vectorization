import numpy as np
import cv2 as cv

from classes.population import Population

img = cv.imread('img/starry-night.jpg')
img = cv.resize(img, (0, 0), fx=.5, fy=.5)

pop_size=10
n_poly=50
n_vertex=3

def genotype(w=(0, img.shape[1]), h=(0, img.shape[0]), c=(0, 256)):
    return np.concatenate([
        np.random.randint(w[0], w[1], (n_poly, n_vertex)),
        np.random.randint(h[0], h[1], (n_poly, n_vertex)),
        np.random.randint(c[0], c[1], (n_poly, 4))
    ], axis=1)

def decode(poly):
    pts = np.array(list(zip(poly[:-3:2], poly[1:-3:2])))
    color, alpha = poly[-4:-1], poly[-1]
    alpha=100
    return pts, color, alpha

def draw(individual):
    result = np.zeros_like(img)
    for poly in individual:
        layer = np.zeros_like(result)
        pts, color, alpha = decode(poly)
        cv.fillPoly(layer, [pts], color.tolist())
        result = cv.addWeighted(result, 1, layer, alpha/512, 0)
    return result

def fitness(ind_img):
    return np.sum((img - ind_img)**2)

population = Population(img)

'''
best = genotype()
gen = 0
while True:
    population = []
    for i in range(pop_size):
        mutation = genotype((-5, 5), (-5, 5), (-10, 10))
        ind = best + mutation
        population.append(ind)

    population.sort(key=lambda x: fitness(draw(x)))
    print(f'Gen {gen}) worst: {fitness(draw(population[-1]))}, best: {fitness(draw(population[0]))}')
    best = population[0]

    cv.imshow('img', img)
    cv.imshow('best', draw(best))
    cv.waitKey(1)
    gen += 1   
'''