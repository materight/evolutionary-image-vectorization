from PIL import Image
import numpy as np
import cv2 as cv
import time

from classes.population import Population

# Set windows properties
cv.namedWindow('result')
# cv.setWindowProperty('result', cv.WND_PROP_TOPMOST, 1)

# Load image
img = Image.open('img/mona-lisa.jpg')
img = np.array(img)
img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
img = cv.resize(img, (0, 0), fx=.5, fy=.5)
print(f'Image size: {img.shape}')

'''
gray_filtered = cv.bilateralFilter(img, 7, 50, 50)
img = cv.Canny(gray_filtered, 80, 120)
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
'''

# Genetic algorithm
population = Population(
    img,
    pop_size=50, 
    n_poly=100, 
    n_vertex=3, 
    selection_cutoff=.1
)

while True:
    start_time = time.time()
    gen, best, others = population.next()
    print(f'{gen}) {round((time.time() - start_time)*1000)}ms, best: {best.fitness}, ({best.n_poly} poly)')
    #cv.imshow('result', np.hstack([img, best.img, *[o.img for o in others]]))
    best_img = cv.resize(best.draw(), img.shape[1::-1])
    cv.imshow('result', np.hstack([img, best_img]))

    cv.waitKey(1)