import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

class Problem:
    RGB = 1
    GRAYSCALE = 2
    
    
    def __init__(self, problem_type, target, internal_resolution):
        self.problem_type = problem_type
        self.scale_factor = internal_resolution / min(target.shape[:2]) if internal_resolution > 0 else 1
        self.set_target(target)


    def set_target(self, target):
        if self.problem_type == self.RGB:
            self.target = cv.resize(target, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        elif self.problem_type == self.GRAYSCALE:
            self.target = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
            self.target = cv.bilateralFilter(self.target, 9, 200, 200) # Apply bilateral filter to improve edge detection
        else:
            raise ValueError(f'Problem type {self.problem_type} not recognized')


