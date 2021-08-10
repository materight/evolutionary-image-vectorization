import numpy as np
import functools
from numba import njit

@njit
def clip(value, min_value, max_value):
    if value < min_value: return min_value
    if value > max_value: return max_value
    return value

@njit
def normal(mean=0, scale=1):
    return np.random.normal() * scale + mean

@njit
def uniform(minv=0, maxv=1):
    return (np.random.rand() * (maxv - minv)) + minv


@njit
def angle(p):
    a = np.arctan2(p[1], p[0])
    if a <= 0: a = a + 2 * np.pi
    return a

@njit
def distance(p1, p2):
    d = [p2[0] - p1[0], p2[1] - p1[1]]
    return d[0]**2 + d[1]**2

@njit
def compute_line_coords(center, rotation, length, center_dist=0):
    r = length / 2 # Line radius
    d = np.array([r * np.cos(rotation), r * np.sin(rotation)]) # Compute displacement of line ends from line center
    cd = np.array([center_dist * np.cos(rotation+np.pi/2), center_dist * np.sin(rotation+np.pi/2)]) # Compute displacement of filters centers from line center (perpendicular to line)
    tcenter = center + cd # Translated center
    return np.concatenate((tcenter + d, tcenter - d))

@njit
def sample_points(line_coords, n_points):
    p1j, p1i, p2j, p2i = line_coords
    return interpolate(np.array([p1i, p1j]), np.array([p2i, p2j]), n_points)

@njit
def interpolate(x1, x2, n_points):
    d = (x2 - x1) / n_points
    interpolations = [x1 + d * i for i in range(0, n_points)]
    return interpolations