from PIL import Image
import numpy as np
import pandas as pd
import cv2 as cv
import os, sys, shutil, time, itertools
import multiprocessing as mp
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from classes.ga.ga import GA
from classes.pso.pso import PSO

RESULTS_BASE_PATH = 'results/benchmark'
IMAGE = 'mona_lisa'
ALGORITHM = GA # GA or PSO
MAX_GENERATIONS = 500

ALGORITHM_PARAMS = {
    GA: dict(
        pop_size=[50, 100],
        n_poly=[50, 100],
        n_vertex=[3, 5],
        selection_cutoff=[.1, .2],
        mutation_chances=[(0.01, 0.01, 0.01)],
        mutation_factors=[(0.2, 0.2, 0.2)]
    ),
    PSO: dict(
        swarm_size=[100, 300],
        neighborhood_size=[5],
        coeffs=[(0.5, 4.1, 0.1)],
        min_distance=[0]
    ),
}

# Create folder for results
if os.path.exists(RESULTS_BASE_PATH):
    shutil.rmtree(RESULTS_BASE_PATH)
os.makedirs(RESULTS_BASE_PATH)

# Generate params list with all possible combinations
keys, values = zip(*ALGORITHM_PARAMS[ALGORITHM].items())
params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Load image
img = cv.cvtColor(np.array(Image.open(f'samples/{IMAGE}.jpg')), cv.COLOR_RGB2BGR)

# Run a single instance of the selected algoritm
def run(run_idx, params):
    alg = ALGORITHM(img, **params)
    times, fitnesses = [], []
    pbar = tqdm(total=MAX_GENERATIONS, desc=f'Run {run_idx}', position=run_idx)
    for i in range(MAX_GENERATIONS):
        start_time = time.time()
        if ALGORITHM is GA:
            gen, best, population = alg.next()
            fitness = best.fitness
        elif ALGORITHM is PSO:
            gen, fitness = alg.next()
        tot_time = round((time.time() - start_time)*1000)
        times.append(tot_time)
        fitnesses.append(fitness)
        sys.stdout.flush()
        pbar.update()
    # Save best individual
    if ALGORITHM is GA:
        best_individual = None#best.draw()
    elif ALGORITHM is PSO:
        best_individual = None#alg.draw()
    pbar.close()
    return times, fitnesses, best_individual

start_time = time.time()
# Launch different subprocesses for each combination
with ProcessPoolExecutor(mp_context=mp.get_context('fork')) as executor:
    futures = [executor.submit(run, i, params) for i, params in enumerate(params_list)] #tqdm(total=MAX_GENERATIONS, position=i)
    results = [future.result() for future in futures]
    times, fitnesses, best_individual = zip(*results)

# Save best generated individuals for each run
'''
TODO
os.makedirs(f'{RESULTS_BASE_PATH}/best_individuals')
for i, img in enumerate(best_individual):
    cv.imwrite(f'{RESULTS_BASE_PATH}/best_individuals/{i}.jpg', img)
'''

# Compute dataframe from results
results = pd.DataFrame.from_dict(params_list)
results['best_fitness'] = [np.min(f) for f in fitnesses]
results['avg_time_ms'] = [np.mean(t, dtype=int) for t in times]
results.to_csv('results/results.csv', sep='\t')

# Compute total time spent
for r in results: print('\n')
tot_time = round((time.time() - start_time)*1000)
print(f'Tot time required: {tot_time}ms')
print(results)

# Plot fitnesses
x = range(MAX_GENERATIONS)
for i, f in enumerate(fitnesses):
    plt.plot(x, f, label=f'run {i}')
plt.legend()
plt.show()