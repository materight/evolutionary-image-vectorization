from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt, matplotlib.colors as mplc
import time, itertools, multiprocessing, concurrent.futures
from tqdm import tqdm

from classes.ga.ga import GA
from classes.pso.pso import PSO

IMAGE = 'img/mona_lisa.jpg'
ALGORITHM = GA # GA or PSO
MAX_GENERATIONS = 10

ALGORITHM_PARAMS = {
    GA: dict(
        pop_size=[50],
        n_poly=[50, 100],
        n_vertex=[3, 4, 5],
        selection_cutoff=[.1],
        mutation_chances=[(0.01, 0.01, 0.01)],
        mutation_factors=[(0.2, 0.2, 0.2)]
    ),
    PSO: dict(
        swarm_size=[100, 300],
        neighborhood_size=[5],
        coeffs = [(0.5, 4.1, 0.1)],
        min_distance=[0]
    ),
}

# Generate params list with all possible combinations
keys, values = zip(*ALGORITHM_PARAMS[ALGORITHM].items())
params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Load image
img = cv.cvtColor(np.array(Image.open(IMAGE)), cv.COLOR_RGB2BGR)

# Run a single instance of the selected algoritm
def run(params):
    alg = ALGORITHM(img, **params)
    for i in range(MAX_GENERATIONS):
        start_time = time.time()
        res = alg.next()
        tot_time = round((time.time() - start_time)*1000)
        if ALGORITHM is GA:
            gen, best, population = res
        elif ALGORITHM is PSO:
            gen = res
        print(f'{i}) {tot_time}ms')
        #pbar.update()
    return start_time

# Launch different subprocesses for each combination
NCPU = 4 # Number of parallel processes to spawn. None to use the available number of core
with concurrent.futures.ProcessPoolExecutor(max_workers=NCPU, mp_context=multiprocessing.get_context('fork')) as executor:
    futures = [executor.submit(run, params, img, None) for i, params in enumerate(params_list)] #tqdm(total=MAX_GENERATIONS, position=i)
    results = [future.result() for future in futures]