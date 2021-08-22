from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import time, sys, os
import sklearn.manifold

from classes.ga.ga import GA
from classes.pso.pso import PSO
from classes.operators import selection, replacement, crossover, velocity_update, topology

# Set windows properties
cv.namedWindow('Result')

# Params
SAMPLE = 'unitn.jpg'
ALGORITHM = PSO  # GA or PSO
INTERPOLATION_SIZE = 5 # Number of interpolated frame to save for PSO results. Set to 1 to disable interpolation
VIDEO_INIT_GEN, VIDEO_FRAME_GEN = 2000, 500 # Number of generations to run for the first and for the other frames, respectively 

if len(sys.argv) > 1:
    SAMPLE = sys.argv[1]
    ALGORITHM = PSO if sys.argv[2] == 'PSO' else GA

sample_name, sample_ext = SAMPLE.split('.')
# Load image or video
if sample_ext in ['jpg', 'jpeg', 'png']:
    isvideo = False
    img = cv.cvtColor(np.array(Image.open(f'samples/{SAMPLE}')), cv.COLOR_RGB2BGR)
    fps = 30
elif sample_ext in ['mp4', 'gif']:
    isvideo = True
    video = cv.VideoCapture(f'samples/{SAMPLE}')
    _, img = video.read()
    frame_count = 0
    fps = video.get(cv.CAP_PROP_FPS)
else:
    raise ValueError(f'File extension "{sample_ext}" not supported')

# Prepare to save result as video
fourcc = cv.VideoWriter_fourcc(*'mp4v')
os.makedirs(f'results/videos', exist_ok=True)
out = cv.VideoWriter(f'results/videos/{ALGORITHM.__name__}_{sample_name}.mp4', fourcc, fps, img.shape[:2][::-1])

# Genetic algorithm
ga = GA(
    img,
    pop_size=100,
    n_poly=120,             
    n_vertex=4,
    selection_strategy=selection.TruncatedSelection(.1), # selection.RouletteWheelSelection(), selection.RankBasedSelection(), selection.TruncatedSelection(.1), selection.TournamentSelection(10)
    replacement_strategy=replacement.CommaReplacement(), # replacement.CommaReplacement(), replacement.PlusReplacement(), replacement.CrowdingReplacement(4)
    crossover_type=crossover.UniformCrossover(),         # crossover.OnePointCrossover(), crossover.UniformCrossover(), crossover.ArithmeticCrossover()
    self_adaptive=False,                                 # Self-adaptetion of mutation step-sizes
    mutation_rates=(0.02, 0.02, 0.02),                   # If self_adaptive is True, not used
    mutation_step_sizes=(0.1, 0.1, 0.1)                  # If self_adaptive is True, not used
)

# Particle swarm optimization
pso = PSO(
    img,
    swarm_size=1000,
    line_length=15,
    velocity_update_rule=velocity_update.Standard(),  # velocity_update.Standard(), velocity_update.FullyInformed(), velocity_update.ComprehensiveLearning()
    neighborhood_topology=topology.StarTopology(),  # topology.DistanceTopology(), topology.RingTopology(), topology.StarTopology()
    neighborhood_size=3,
    coeffs=(0.1, 1.5, 1.5),  # Inertia (0.7 - 0.8), cognitive coeff/social coeff (1.5 - 1.7)
    min_distance=2,
    max_velocity=50
)

fbest, favg, fworst = [], [], []
diversities, dist = [], None
try: # Press ctrl+c to exit loop
    print(f'\nRunning {ALGORITHM.__name__} algorithm over "{SAMPLE}".\nPress ctrl+c to terminate the execution.\n')
    while True:
        start_time = time.time()

        # Compute next generation
        additional_info = ''
        if ALGORITHM is GA:
            gen, population = ga.next()
            best = population[0]
            additional_info = f' ({best.fitness_perc * 100:.2f}%), polygons: {best.n_poly}'
            fitness = best.fitness
            fbest.append(best.fitness)
            favg.append(np.mean([i.fitness for i in population]))
            fworst.append(population[-1].fitness)
            if gen % 20 == 0: # Measure diversity every 20 generations
                dist = ga.diversity()
                diversity = dist.sum()
                diversities.append(diversity)
                additional_info += f', diversity: {int(diversity)}'
        elif ALGORITHM is PSO:
            gen, fitness = pso.next()
            fbest.append(fitness)

        # Print and save result
        tot_time = round((time.time() - start_time)*1000)
        print(f'{gen:04d}) {tot_time:04d}ms, fitness: {fitness:.4f}{additional_info}')

        # Obtain current best solution
        if ALGORITHM is GA:
            best_img = best.draw()
        elif ALGORITHM is PSO:
            best_img = pso.draw()

        # Show current best
        result = cv.resize(best_img, img.shape[1::-1])
        result = np.hstack([img, result])
        result = cv.resize(result, None, fx=.6, fy=.6)
        cv.imshow('Result', result)

        # Save result in video
        if (ALGORITHM is GA and ( \
                (isvideo and gen%VIDEO_FRAME_GEN==0) or \
                (not isvideo and gen%10==0))) \
        or (ALGORITHM is PSO):
            if ALGORITHM is GA: frames = [best_img.copy()]
            elif ALGORITHM is PSO: frames = pso.draw_interpolated(INTERPOLATION_SIZE) # Interpolate frames for better visualization
            for frame in frames:
                # frame = cv.putText(frame, f'{gen}', (2, 16), cv.FONT_HERSHEY_PLAIN, 1.4, (0, 0, 255), 2) # Print generation number
                out.write(frame)

        # Key press
        key = cv.waitKey(1) & 0xFF
        if key == ord(' '):
            cv.waitKey(0)

        # Update the target, in case of video input
        if isvideo and ((frame_count==0 and gen>VIDEO_INIT_GEN) or (frame_count>0 and gen%VIDEO_FRAME_GEN==0)): # Optimize over new frame every 100 generations. First frame used for 1000 generations
            ret, img = video.read()
            if not ret: break
            frame_count += 1
            if ALGORITHM is GA: ga.update_target(img)
            elif ALGORITHM is PSO: pso.update_target(img)
        
except KeyboardInterrupt:
    pass

# Save final individual image
os.makedirs(f'results/images', exist_ok=True)
cv.imwrite(f'results/images/{ALGORITHM.__name__}_{sample_name}.jpg', best_img)

# Clear all
cv.destroyAllWindows()
out.release()


# Plots

# Fitness plots
fig, ax = plt.subplots()
fig.suptitle('Fitness trends')
x = range(len(fbest))
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
    ax.plot(range(len(diversities)), diversities, c='b', label='diversity')
    ax.legend()

if dist is not None:
    fig, ax = plt.subplots()
    fig.suptitle('Diversity scatter plot')
    dist_proj = sklearn.manifold.TSNE(metric='precomputed', perplexity=7, random_state=0).fit_transform(dist)
    ax.scatter(dist_proj[:, 0], dist_proj[:, 1], s=4, c='r')

plt.show()
