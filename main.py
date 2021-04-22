import numpy as np
import cv2 as cv

from classes.population import Population

img = cv.imread('img/mona-lisa.jpg')
img = cv.resize(img, (0, 0), fx=.4, fy=.4)
print(f'Image size: {img.shape}')


population = Population(img)

while True:
    gen, best = population.next()
    print(f'{gen}) best: {best.fitness()}')
        
    cv.imshow('img', img)
    cv.imshow('best', best.draw())
    cv.waitKey(1)