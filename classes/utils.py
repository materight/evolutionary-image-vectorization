import numpy as np
from numba import njit

@njit
def clip(value, min_value, max_value):
    if value < min_value: return min_value
    if value > max_value: return max_value
    return value

@njit
def normal(mean=0, scale=1):
    return np.random.normal() * scale + mean