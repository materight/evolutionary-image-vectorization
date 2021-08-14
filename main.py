from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import time

from classes.operators import selection, replacement
from classes.ga.ga import GA
from classes.ga.individual import Individual
from classes.pso.pso import PSO
from classes.pso.particle import Particle

# TODO:
# - Fix self-adaptive strategy
# - finish speciation for GA
# - complete support for videos
# - complete benchmark and run experiments (save plots for speciation and fitness)
# - save images to add to paper

# Set windows properties
cv.namedWindow('Result')

# Load image
IMAGE = 'mona_lisa'
ALGORITHM = GA  # GA or PSO
INTERPOLATION_SIZE = 5 # Number of interpolated frame to save for PSO results. 1 to disable interpolation
img = cv.cvtColor(np.array(Image.open(f'samples/{IMAGE}.jpg')), cv.COLOR_RGB2BGR)

# Save result as video
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(f'results/{ALGORITHM.__name__}_{IMAGE}.mp4', fourcc, 30, img.shape[:2][::-1])

# Genetic algorithm
ga = GA(
    img,
    pop_size=100,
    n_poly=100,  # Initialize individuals with different numbers of polygons
    n_vertex=3,
    selection_strategy=selection.TruncatedSelection(.1),  # selection.RouletteWheelSelection(), selection.RankBasedSelection(), selection.TruncatedSelection(.1), selection.TournamentSelection(10)
    replacement_strategy=replacement.CommaReplacement(), # replacement.CommaReplacement(), replacement.PlusReplacement()
    crossover_type=Individual.UNIFORM_CROSSOVER,  # Individual.ONE_POINT_CROSSOVER, Individual.UNIFORM_CROSSOVER, Individual.ARITHMETIC_CROSSOVER, Individual.ALIGNED_CROSSOVER
    self_adaptive=True,
    mutation_chances=(0.02, 0.02, 0.02), 
    mutation_factors=(0.2, 0.2, 0.2), # If evolution strategies is true, not used
    niche_size=0  # 0.001, 0
)

# Particle swarm optimization
pso = PSO(
    img,
    swarm_size=500,
    line_length=20,
    velocity_update_rule=Particle.STANDARD,  # Particle.STANDARD, Particle.FULLY_INFORMED, Particle.COMPREHENSIVE_LEARNING
    neighborhood_topology=Particle.DISTANCE_TOPOLOGY,  # Particle.DISTANCE_TOPOLOGY, Particle.RING_TOPOLOGY, Particle.STAR_TOPOLOGY
    neighborhood_size=3,
    coeffs=(0.1, 1.7, 1.5),  # Inertia (0.7 - 0.8), cognitive coeff/social coeff (1.5 - 1.7) # Check https://doi.org/10.1145/1830483.1830492
    min_distance=10,
    max_velocity=10
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
        additional_info = f' ({best.fitness_perc * 100:.2f}%), polygons: {best.n_poly}'
        fitness = best.fitness
        fbest.append(best.fitness)
        favg.append(np.mean([i.fitness for i in population]))
        fworst.append(population[-1].fitness)
        if diversity is not None:
            diversities.append(diversity)
            additional_info += f', diversity: {diversity}'
    elif ALGORITHM is PSO:
        gen, fitness = pso.next()
        fbest.append(fitness)

    # Print and save result
    tot_time = round((time.time() - start_time)*1000)
    print(f'{gen:04d}) {tot_time:03d}ms, fitness: {fitness:.2f}{additional_info}')

    # Obtain current best solution
    if ALGORITHM is GA:
        best_img = best.draw()
    elif ALGORITHM is PSO:
        best_img = pso.draw()

    # Show current best
    best_img = cv.resize(best_img, img.shape[1::-1])
    result = np.hstack([img, best_img])
    result = cv.resize(result, None, fx=.6, fy=.6)
    cv.imshow('Result', result)

    # Save result in video
    if (ALGORITHM is GA and gen % 10 == 0) or (ALGORITHM is PSO):
        if ALGORITHM is GA: frames = [best_img.copy()]
        elif ALGORITHM is PSO: frames = pso.draw_interpolated(INTERPOLATION_SIZE) # Interpolate frames for better visualization
        for frame in frames:
            out_frame = cv.putText(frame, f'{gen}', (2, 16), cv.FONT_HERSHEY_PLAIN, 1.4, (0, 0, 255), 2)
            out.write(out_frame)

    # Key press
    key = cv.waitKey(10) & 0xFF
    if key == ord(' '):
        cv.waitKey(0)
    elif key == ord('q'):
        break

    # Update the target, in case the algorithm is in real-time
    # ga.update_target(img)
    # pso.update_target(img)

cv.destroyAllWindows()
out.release()


# Plots
x = range(len(fbest))

# Fitness plots
fig, ax = plt.subplots()
fig.suptitle('Fitness trends')
ax.plot(x, fbest, c='r', label='best')
if len(favg) > 0:
    ax.plot(x, favg, c='b', label='average')
if len(fworst) > 0:
    ax.plot(x, fworst, c='g', label='worst')
ax.legend()

# Diversity plots
if len(diversities) > 0:
    fig, ax = plt.subplots()
    fig.suptitle('Diversity')
    ax.plot(x, diversities, c='b', label='diversity')
    ax.legend()

plt.show()
