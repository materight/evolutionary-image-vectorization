from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time


from classes.ga.ga import GA
from classes.pso.pso import PSO

# Set windows properties
cv.namedWindow('Result')

# Load image
img = Image.open('img/mona-lisa.jpg')
img = np.array(img)
img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
img = cv.resize(img, (0, 0), fx=.6, fy=.6)
print(f'Image size: {img.shape}')


# Genetic algorithm
ea = GA(
    img,
    pop_size=50,
    n_poly=50,
    n_vertex=3,
    selection_cutoff=.1,
    mutation_chances=(0.01, 0.01, 0.01),
    mutation_factors=(0.2, 0.2, 0.2),
    internal_resolution=75 # -1 to use original size
)

# Particle swarm optimization
pso = PSO(
    img,
    swarm_size=100,
    internal_resolution=75 # -1 to use original size
)

hbest, havg, hworst = [], [], []
while True:
    start_time = time.time()
    
    '''
    gen, best, population = ea.next()
    print(f'{gen}) {round((time.time() - start_time)*1000)}ms, best: {best.fitness}, ({best.n_poly} poly)')
    hbest.append(best.fitness)
    havg.append(np.average([ind.fitness for ind in population]))
    hworst.append(population[-1].fitness)

    best_img = cv.resize(best.draw(), img.shape[1::-1])
    cv.imshow('Result', np.hstack([img, best_img]))
    '''

    iteration = pso.next()
    print(f'{iteration})')
    best_img = cv.resize(pso.draw(), img.shape[1::-1])
    cv.imshow('Result', np.hstack([img, best_img]))
    
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        run = False
        break 
    
    #ea.update_target(img)

cv.destroyAllWindows()
x = range(len(hbest))
plt.plot(x, hbest, c='r', label='best')
#plt.plot(x, havg, c='g', label='avg')
#plt.plot(x, hworst, c='k', label='worst')
plt.legend()
plt.show()