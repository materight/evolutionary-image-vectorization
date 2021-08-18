import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from classes.ga.ga import GA
from classes.pso.pso import PSO

'''
This script can be used to analyze the results obtained by running benchmark.py
'''

ALGORITHM = GA # GA or PSO
RESULTS_BASE_PATH = f'results/benchmark/{ALGORITHM.__name__}'

# Read data
exp, prog = [], [] # Experiments info and progression by generation data
for exp_path in glob.glob(f'{RESULTS_BASE_PATH}/*'):
    e = pd.read_csv(f'{exp_path}/results.csv', index_col=0, header=None, squeeze=True)
    p = pd.read_csv(f'{exp_path}/progress.csv').assign(run_idx=e.run_idx, rep_idx=e.rep_idx)
    exp.append(e)
    prog.append(p)
exp = pd.concat(exp, ignore_index=True, axis=1).T
prog = pd.concat(prog, ignore_index=True)

# Set correct dtypes
exp = exp.astype(dict(run_idx=int, rep_idx=int, pop_size=int, n_poly=int, n_vertex=int, self_adaptive=bool, fitness=float, exec_time=float))
prog = prog.astype(dict(run_idx=int, rep_idx=int))

# Filter out bad experiments
FITNESS_THRESH = 5000
prog = prog[~prog.run_idx.isin(exp[exp.fitness > FITNESS_THRESH].run_idx)] # Remove experiments with final fitness below this thresh
exp = exp[exp.fitness <= FITNESS_THRESH]

# Plots
# Fitness convergence
plt_fitness = prog.pivot(index='generation', columns='run_idx', values='best_fitness')
plt_fitness.plot()

# Diversity
plt_diversity = prog[~prog.diversity.isna()]
plt_diversity = plt_diversity.pivot(index='generation', columns='run_idx', values='diversity')
highlight = ('replacement_strategy', lambda x: x.str.contains('CrowdingReplacement')) # or None
color = np.where(exp.set_index('run_idx')[highlight[0]].pipe(highlight[1]).loc[plt_diversity.columns.to_list()], 'r', 'b')
plt_diversity.plot(color=color)

# Scatter for Pareto front
ax = exp.plot.scatter('n_vertex', 'fitness')
exp.groupby('n_vertex').fitness.min().reset_index().plot.scatter('n_vertex', 'fitness', c='r', ax=ax) # Plot max in red

# Show all
plt.show()