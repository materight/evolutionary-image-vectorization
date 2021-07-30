from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt, matplotlib.colors as mplc
import time

from classes import selection
from classes.ga.ga import GA
from classes.ga.individual import Individual
from classes.pso.pso import PSO
from classes.pso.particle import Particle

# Set windows properties
cv.namedWindow('Result')

# Load image
IMAGE = 'starry_night'
ALGORITHM = GA # GA or PSO
img = cv.cvtColor(np.array(Image.open(f'samples/{IMAGE}.jpg')), cv.COLOR_RGB2BGR)

# Save result as video
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(f'results/{ALGORITHM.__name__}_{IMAGE}.mp4', fourcc, 30, img.shape[:2][::-1])

# Genetic algorithm
ga = GA(
    img,
    pop_size=50,
    n_poly=150, # Initialize individuals with different numbers of polygons
    n_vertex=4,
    selection_strategy=selection.TruncatedSelection(.1),  # selection.RouletteWheelSelection(), selection.RankBasedSelection(), selection.TruncatedSelection(.1), selection.TournamentSelection(10)
    crossover_type=Individual.UNIFORM_CROSSOVER,  # Individual.ONE_POINT_CROSSOVER, Individual.UNIFORM_CROSSOVER, Individual.FITNESS_PROPORTIONAL_CROSSOVER
    mutation_chances=(0.01, 0.01, 0.01),
    mutation_factors=(0.2, 0.2, 0.2),
    niche_size=0 # 0.001, 0
)

# Particle swarm optimization
pso = PSO(
    img,
    swarm_size=500,
    velocity_update_rule=Particle.STANDARD,  # Particle.STANDARD, Particle.FULLY_INFORMED, Particle.COMPREHENSIVE_LEARNING
    neighborhood_topology=Particle.DISTANCE_TOPOLOGY,  # Particle.DISTANCE_TOPOLOGY, Particle.RING_TOPOLOGY, Particle.STAR_TOPOLOGY
    neighborhood_size=3,
    coeffs=(0.1, 1.5, 1.5), # Inertia (0.7 - 0.8), cognitive coeff, social coeff (1.5 - 1.7) # Check https://doi.org/10.1145/1830483.1830492
    min_distance=10,
    max_velocity=20
)

fbest, favg, fworst = [], [], [] 
diversities = []
while True:
    start_time = time.time()

    # Compute next generation
    additional_info = ''
    if ALGORITHM is GA: 
        gen, population, diversity = ga.next()
        best = population[0]  
        fitness = best.fitness     
        fbest.append(best.fitness)
        favg.append(np.mean([i.fitness for i in population]))
        fworst.append(population[-1].fitness)
        if diversity is not None: 
            diversities.append(diversity)
        additional_info = f', polygons: {best.n_poly}'
    elif ALGORITHM is PSO: 
        gen, fitness = pso.next()
        fbest.append(fitness)

    # Print and save result
    tot_time = round((time.time() - start_time)*1000)
    print(f'{gen:04d}) {tot_time:03d}ms, fitness: {fitness}{additional_info}')

    # Obtain current best solution
    target_img = None
    if ALGORITHM is GA: 
        best_img = best.draw()
    elif ALGORITHM is PSO:
        target_img = np.log(pso.problem.target.copy()+1)
        target_img = cv.normalize(target_img, None, 0, 255, norm_type=cv.NORM_MINMAX)
        target_img = cv.cvtColor(target_img.astype(np.uint8), cv.COLOR_GRAY2BGR)
        best_img = pso.draw()
        target_img = np.where(best_img[:,:] == [255,255,255], [0,0,255], target_img[:,:]).astype(np.uint8)
    
    # Show current best
    best_img = cv.resize(best_img, img.shape[1::-1])
    result = np.hstack([img, best_img]) 
    if target_img is not None: result = np.hstack([result, target_img])
    result = cv.resize(result, None, fx=.6, fy=.6) # target_img 
    cv.imshow('Result', result) 
    
    # Save result in video
    if gen % 5 == 0:
        out_frame = cv.putText(best_img.copy(), f'{gen}', (2,16), cv.FONT_HERSHEY_PLAIN, 1.3, (255,255,255), 2)
        out.write(out_frame)

    # Key press
    key = cv.waitKey(1) & 0xFF
    if key == ord(' '): cv.waitKey(0) 
    elif key == ord('q'): break 
    
    # Update the target, in case the algorithm is in real-time
    #ga.update_target(img)
    #pso.update_target(img)

cv.destroyAllWindows()
out.release()


# Plots
x = range(len(fbest))

# Fitness plots
fig, ax = plt.subplots()
fig.suptitle('Fitness trends')
ax.plot(x, fbest, c='r', label='best')
if len(favg) > 0: ax.plot(x, favg, c='b', label='average')
if len(fworst) > 0: ax.plot(x, fworst, c='g', label='worst')
ax.legend()

# Diversity plots
if len(diversities) > 0:
    fig, ax = plt.subplots()
    fig.suptitle('Diversity')
    ax.plot(x, diversities, c='b', label='diversity')
    ax.legend()

plt.show()