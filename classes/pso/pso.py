import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw

from ..problem import Problem
from .particle import Particle

class PSO:
    def __init__(self, target, swarm_size=50, internal_resolution=-1):
        self.iteration = 0
        self.problem = Problem(Problem.PSO, target, internal_resolution)
        self.swarm = []
        for i in range(swarm_size):
            self.swarm.append(Particle.random(self.problem))
        self.swarm.sort(key=lambda i: i.fitness)

    def next(self):
        self.iteration += 1

        for particle in self.swarm:
            particle.move(self.swarm)

        return self.iteration

    def draw(self):
        scale = 1/self.problem.scale_factor  # Rescale internal image target to full scale
        img = Image.new('RGB', (int(self.problem.target.shape[1]*scale), int(self.problem.target.shape[0]*scale)), color='white')
        draw = ImageDraw.Draw(img, 'RGB')
        for particle in self.swarm:
            draw.line(tuple(particle.line.coords*scale), fill=(0,0,0), width=int(2*scale))
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        return img

    def update_target(self, target):
        self.problem.set_target(target)
    

