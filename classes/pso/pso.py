import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw
import random

from ..problem import Problem
from .particle import Particle
from ..utils import interpolate

class PSO:

    def __init__(self, target, swarm_size, line_length, velocity_update_rule, neighborhood_topology, neighborhood_size, coeffs, min_distance, max_velocity, internal_resolution=-1):
        self.iteration = 0
        self.problem = Problem(Problem.GRAYSCALE, target, internal_resolution)
        self.swarm_size = swarm_size
        self.line_length = line_length
        self.velocity_update_rule = velocity_update_rule
        self.neighborhood_topology = neighborhood_topology
        self.neighborhood_size = neighborhood_size
        self.coeffs = coeffs
        self.min_distance = min_distance
        self.max_velocity = max_velocity
        self.swarm = []
        for idx in range(swarm_size):
            self.swarm.append(Particle.random(self.problem, idx, line_length, self.max_velocity))

    def next(self):
        self.iteration += 1
        fitness = 0
        for i, particle in enumerate(self.swarm):
            particle.move(i, self.swarm, self.velocity_update_rule, self.neighborhood_topology, self.neighborhood_size, self.coeffs, self.min_distance, self.max_velocity)
            fitness += particle.fitness
        fitness = fitness / len(self.npswarm)
        return self.iteration, fitness

    def draw(self):
        return PSO._draw_npswarm(self.npswarm, self.problem.scale_factor, self.problem.target.shape)

    def draw_interpolated(self, prev_npswarm, n_points):
        imgs = []
        for npswarm in interpolate(prev_npswarm, self.npswarm, n_points):
            imgs.append(PSO._draw_npswarm(npswarm, self.problem.scale_factor, self.problem.target.shape))
        return imgs

    def _draw_npswarm(npswarm, scale, img_size):
        scale = 1 / scale  # Rescale internal image target to full scale
        img = Image.new('RGB', (int(img_size[1]*scale), int(img_size[0]*scale)), color='white')
        draw = ImageDraw.Draw(img, 'RGB')
        for coords in npswarm:
            draw.line(tuple(coords*scale), fill=(0,0,0), width=int(1*scale))
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        return img

    def update_target(self, target):
        self.problem.set_target(target)
        self.swarm = []
        for idx in range(self.swarm_size):
            self.swarm.append(Particle.random(self.problem, idx, self.line_length, self.max_velocity))

    @property
    def npswarm(self): # Return curent swarm as numpy array
        return np.stack([p.line.coords for p in self.swarm])

