from PIL import Image
import numpy as np
import pandas as pd
import cv2 as cv
import os
import sys
import shutil
import time
import itertools
import random
from tqdm import tqdm

from classes.ga.ga import GA
from classes.pso.pso import PSO
from classes.operators import selection, replacement, crossover, velocity_update, topology

'''
This script can be used to automatically test different hyperparameters conbinations and reports the obtained results.
'''

random.seed(0)

SAMPLE = 'mona_lisa.jpg'
ALGORITHM = PSO  # GA or PSO
MAX_GENERATIONS = 1000 if ALGORITHM is GA else 50
NUM_REPETITONS = 1  # Number of repetitions of each run
RESULTS_BASE_PATH = f'results/benchmark/{ALGORITHM.__name__}'

ALGORITHM_PARAMS = {
    GA: dict(
        pop_size=[50, 100],
        n_poly=[50, 100, 200],
        n_vertex=[5],
        selection_strategy=[selection.RouletteWheelSelection(), selection.TruncatedSelection(.1), selection.TournamentSelection(10), selection.TruncatedSelection(.2)],
        replacement_strategy=[replacement.CommaReplacement(), replacement.PlusReplacement(), replacement.CrowdingReplacement(2), replacement.CrowdingReplacement(5)],
        crossover_type=[crossover.OnePointCrossover(), crossover.UniformCrossover()],
        self_adaptive=[False, True],
        mutation_rates=[(0.02, 0.02, 0.02)],
        mutation_step_sizes=[(0.2, 0.2, 0.2)]
    ),
    PSO: dict(
        swarm_size=[500, 1000],
        line_length=[10, 20],
        velocity_update_rule=[velocity_update.Standard(), velocity_update.FullyInformed(), velocity_update.ComprehensiveLearning()],
        neighborhood_topology=[topology.DistanceTopology(), topology.StarTopology(), topology.RingTopology()],
        neighborhood_size=[3, 5],
        coeffs=[(0.1, 1.7, 1.2), (0.7, 1.5, 1.5)],
        min_distance=[0, 5, 10],
        max_velocity=[50]
    ),
}

# Create folder for results
os.makedirs(RESULTS_BASE_PATH, exist_ok=True)

# Generate params list with all possible combinations
keys, values = zip(*ALGORITHM_PARAMS[ALGORITHM].items())
params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
random.shuffle(params_list)

# Load image
img = cv.cvtColor(np.array(Image.open(f'samples/{SAMPLE}')), cv.COLOR_RGB2BGR)

# Convert params dict to string


def dict_to_str(dict, sep):
    return sep.join([f'{k}={v}' for k, v in dict.items()])

# Run a single instance of the selected algoritm


def run(run_idx, rep_idx, params):
    run_name = f'run{run_idx}-rep{rep_idx}){dict_to_str(params, ",")}'
    EXP_PATH = f'{RESULTS_BASE_PATH}/{run_name}'
    if not os.path.exists(EXP_PATH):  # Do not repeat experiments if the results are already available
        alg = ALGORITHM(img, **params)
        exec_times, fbest, favg, fworst, diversities, fbest_perc = [], [], [], [], [], []
        pbar = tqdm(range(MAX_GENERATIONS))
        for i in pbar:
            start_time = time.time()
            if ALGORITHM is GA:
                gen, population = alg.next()
                best = population[0]
                fitness = best.fitness
                fbest.append(fitness)
                fbest_perc.append(best.fitness_perc*100)
                favg.append(np.mean([i.fitness for i in population]))
                fworst.append(population[-1].fitness)
                diversities.append(alg.diversity().sum() if gen % 20 == 0 else None)  # Measure diversity every 20 generations
            elif ALGORITHM is PSO:
                gen, fitness = alg.next()
                fbest.append(fitness)
            # Compute total time spent
            tot_time = round((time.time() - start_time)*1000)
            exec_times.append(tot_time)
            pbar.set_description(f'Fitness: {fitness:.2f}')
        # Compute image of final best individual
        if ALGORITHM is GA:
            best_img = best.draw()
        elif ALGORITHM is PSO:
            best_img = alg.draw()
        # Save best image
        os.makedirs(EXP_PATH)
        cv.imwrite(f'{EXP_PATH}/best.jpg', best_img)
        # Save convergence information for each generation
        data = {'execution_time_ms': exec_times, 'best_fitness': fbest}
        if ALGORITHM is GA:
            data = {'execution_time_ms': exec_times, 'best_fitness_perc': fbest_perc, 'best_fitness': fbest, 'avg_fitness': favg, 'worst_fitness': fworst, 'diversity': diversities}
        elif ALGORITHM is PSO:
            data = {'execution_time_ms': exec_times, 'best_fitness': fbest}
        progress = pd.DataFrame(data, index=range(1, len(fbest) + 1))
        progress.index.name = 'generation'
        progress.to_csv(f'{EXP_PATH}/progress.csv')
        # Save final optimization results
        results = pd.DataFrame.from_dict({'run_idx': run_idx, 'rep_idx': rep_idx, **params, 'fitness': fbest[-1], 'exec_time': np.mean(exec_times)}, orient='index')
        results.to_csv(f'{EXP_PATH}/results.csv', header=False)


def merge_results():
    pass


# Execute experiments
print(f'Total number of experiments: {len(params_list)}\n')
for run_idx, params in enumerate(params_list, start=1):
    for rep_idx in range(1, NUM_REPETITONS + 1):
        try:
            print(f'Run {run_idx}/{len(params_list)} rep {rep_idx}/{NUM_REPETITONS} with params:')
            print('\t', dict_to_str(params, '\n\t '))
            run(run_idx, rep_idx, params)
        except Exception as e:
            print(f'Exception at run {run_idx} and rep {rep_idx}:\n', e)
