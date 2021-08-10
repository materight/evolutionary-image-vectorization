import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw

from ..problem import Problem
from .particle import Particle

class PSO:

    def __init__(self, target, swarm_size=100, velocity_update_rule=Particle.STANDARD, neighborhood_topology=Particle.STAR_TOPOLOGY, neighborhood_size=5, coeffs=(0.5, 4.1, 0.1), min_distance=5, max_velocity=-1, internal_resolution=-1):
        self.iteration = 0
        self.problem = Problem(Problem.GRAYSCALE, target, internal_resolution)
        self.velocity_update_rule = velocity_update_rule
        self.neighborhood_topology = neighborhood_topology
        self.neighborhood_size = neighborhood_size
        self.coeffs = coeffs
        self.min_distance = min_distance
        self.max_velocity = max_velocity
        self.swarm = []
        for i in range(swarm_size):
            self.swarm.append(Particle.random(self.problem, self.max_velocity))


    def next(self):
        self.iteration += 1
        fitness = 0
        for i, particle in enumerate(self.swarm):
            particle.move(i, self.swarm, self.velocity_update_rule, self.neighborhood_topology, self.neighborhood_size, self.coeffs, self.min_distance, self.max_velocity)
            fitness += particle.fitness
        return self.iteration, fitness


    def draw(self):
        scale = 1/self.problem.scale_factor  # Rescale internal image target to full scale
        img = Image.new('RGB', (int(self.problem.target.shape[1]*scale), int(self.problem.target.shape[0]*scale)), color='white')
        draw = ImageDraw.Draw(img, 'RGB')
        for i, particle in enumerate(self.swarm):
            draw.line(tuple(particle.line.coords*scale), fill=(0,0,0), width=2*int(scale))
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        return img


    def update_target(self, target):
        self.problem.set_target(target)
