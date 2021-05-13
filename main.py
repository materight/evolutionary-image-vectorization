from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt, matplotlib.colors as mplc
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
    mutation_factors=(0.2, 0.2, 0.2)
)

# Particle swarm optimization
pso = PSO(
    img,
    swarm_size=300,
    neighborhood_size=3,
    coeffs = (0.5, 4.1, 0.1), # Inertia, cognitive coeff, social coeff
    min_distance=0
)



hbest, havg, hworst = [], [], []
while True:
    start_time = time.time()
    
    '''
    gen, best, population = ea.next()
    
    tot_time = round((time.time() - start_time)*1000)
    print(f'{gen}) {tot_time}ms, best: {best.fitness}, ({best.n_poly} poly)')
    hbest.append(best.fitness)
    havg.append(np.average([ind.fitness for ind in population]))
    hworst.append(population[-1].fitness)

    best_img = cv.resize(best.draw(), img.shape[1::-1])
    cv.imshow('Result', np.hstack([img, best_img]))
    
    '''

    iteration = pso.next()

    tot_time = round((time.time() - start_time)*1000)
    print(f'{iteration}) {tot_time}ms')
    target_img = np.log(pso.problem.target+1)
    target_img = cv.normalize(target_img, None, 0, 255, norm_type=cv.NORM_MINMAX)
    target_img = cv.cvtColor(target_img.astype(np.uint8), cv.COLOR_GRAY2BGR)
    
    best_img = cv.resize(pso.draw(), img.shape[1::-1])
    
    # best_img = np.where(best_img == 0, 255, 0).astype(np.uint8) # Invert colors
    target_img = np.where(best_img[:,:] == [255,255,255], [0,0,255], target_img[:,:]).astype(np.uint8)
    cv.imshow('Result', np.hstack([img, target_img, best_img]))
    
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break 
    
    #ea.update_target(img)
    #pso.update_target(img)

cv.destroyAllWindows()
x = range(len(hbest))
plt.plot(x, hbest, c='r', label='best')
#plt.plot(x, havg, c='g', label='avg')
#plt.plot(x, hworst, c='k', label='worst')
plt.legend()
plt.show()