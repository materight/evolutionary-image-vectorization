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
    return d[0]*d[0] + d[1]*d[1]

@njit
def clockwise_compare(p1, p2):
    a1, a2 = angle(p1), angle(p2)
    if a1 < a2: return 1
    d1, d2 = distance((0,0), p1), distance((0,0), p2)
    if a1 == a2 and d1 < d2:
        return 1
    return -1  

def clockwise_sort(pts):
    center = [0, 0]
    for p in pts:
        center[0], center[1] = center[0] + p[0], center[1] + p[1]
    center[0], center[1] = center[0]/len(pts), center[1]/len(pts)
    for p in pts:
        p[0], p[1] = p[0] - center[0], p[1] - center[1]
    # Sort
    pts.sort(key=functools.cmp_to_key(clockwise_compare))
    for p in pts:
        p[0], p[1] = int(p[0] + center[0]), int(p[1] + center[1])
    return pts
