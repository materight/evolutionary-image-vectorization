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
BENCHMARK_BASE_PATH = f'results/benchmark/{ALGORITHM.__name__}'
RESULTS_BASE_PATH = f'results/plots/{ALGORITHM.__name__}'
os.makedirs(RESULTS_BASE_PATH, exist_ok=True)

# Read data
df = [] # Experiments info and progression by generation data
for i, exp_path in enumerate(glob.glob(f'{BENCHMARK_BASE_PATH}/*')):
    e = pd.read_csv(f'{exp_path}/results.csv', index_col=0, header=None, squeeze=True).rename({'fitness': 'final_fitness'})
    p = pd.read_csv(f'{exp_path}/progress.csv')
    row = p.assign(id=i, **e)
    row = row[['id', *e.index, *p.columns]] # re-order columns
    df.append(row)
df = pd.concat(df, ignore_index=True)
df = df.astype(dict(run_idx=int, rep_idx=int, pop_size=int, n_poly=int, n_vertex=int, self_adaptive=bool, final_fitness=float, exec_time=float))

if ALGORITHM is GA:
    # Filter out bad experiments
    FITNESS_THRESH = 5000
    df = df[df.final_fitness <= FITNESS_THRESH] # Remove experiments with final fitness below this thresh

    # Fitness convergence
    plt_fitness = df.pivot(index='generation', columns='id', values='best_fitness')
    ax = plt_fitness.plot(legend=False)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    ax.set_xlim(plt_fitness.index.min(), plt_fitness.index.max())
    plt.tight_layout()
    plt.savefig(f'{RESULTS_BASE_PATH}/fitness.jpg', dpi=200, bbox_inches='tight')
    
    # Diversity
    plt_diversity = df[~df.diversity.isna()]
    plt_diversity = plt_diversity.assign(crowding=plt_diversity.replacement_strategy.str.contains('CrowdingReplacement'))
    plt_diversity = plt_diversity.pivot(index='generation', columns=['crowding', 'id'], values='diversity')
    plt_diversity_mean = plt_diversity.groupby('crowding', axis=1).mean() 
    plt_diversity_std = plt_diversity.groupby('crowding', axis=1).std() / 2
    plt_diversity_low, plt_diversity_high = plt_diversity_mean - plt_diversity_std, plt_diversity_mean + plt_diversity_std
    plt.figure()
    ax = plt_diversity_mean[True].rename('Crowding replacement').plot(c='tab:red', legend=True)
    plt_diversity_mean[False].rename('Other replacements').plot(c='tab:blue', legend=True, ax=ax)
    ax.fill_between(plt_diversity_mean.index, plt_diversity_low[False], plt_diversity_high[False], color='tab:blue', alpha=.2)
    ax.fill_between(plt_diversity_mean.index, plt_diversity_low[True], plt_diversity_high[True], color='tab:red', alpha=.2)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Diversity')
    ax.set_xlim(plt_diversity_mean.index.min(), plt_diversity_mean.index.max())
    plt.tight_layout()
    plt.savefig(f'{RESULTS_BASE_PATH}/diversity.jpg', dpi=200, bbox_inches='tight')

    # Scatter for Pareto front
    plt_scatter = df.groupby('id')[['n_vertex', 'final_fitness']].first()
    plt_pareto_front = plt_scatter.groupby('n_vertex', as_index=False)['final_fitness'].min()
    ax = plt_scatter.plot.scatter('n_vertex', 'final_fitness', c='tab:blue', s=50)
    plt_pareto_front.plot.scatter('n_vertex', 'final_fitness', c='tab:red', s=50, ax=ax) # Plot min fitness in red
    ax.set_xlabel('Number of verices')
    ax.set_ylabel('Fitness')
    ax.set_xticks(range(plt_pareto_front.n_vertex.min() - 1, plt_pareto_front.n_vertex.max() + 2))
    plt.tight_layout()
    plt.savefig(f'{RESULTS_BASE_PATH}/pareto_front.jpg', dpi=200, bbox_inches='tight')
else:
    pass

# Save plots


# Show plots
plt.show()
