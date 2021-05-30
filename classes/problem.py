import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

class Problem:
    RGB = 1
    EDGES = 2
    
    def __init__(self, problem_type, target, internal_resolution):
        self.problem_type = problem_type
        self.scale_factor = internal_resolution / min(target.shape[:2]) if internal_resolution > 0 else 1
        self.set_target(target)


    def set_target(self, target):
        if self.problem_type == self.RGB:
            self.target = cv.resize(target, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        elif self.problem_type == self.EDGES:
            gray_filtered = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
            gray_filtered = cv.bilateralFilter(gray_filtered, 9, 80, 50)
            # Extract image contours
            self.target = cv.Canny(gray_filtered, 140, 150)
            self.target = np.where(self.target > 0, 0, 255).astype(np.uint8)
            # Compute distance-based fitness landscape
            self.target = cv.distanceTransform(self.target, cv.DIST_C, 3)
            #self.target = np.log(self.target+1)
            self.target = cv.normalize(self.target, None, 0, 255, cv.NORM_MINMAX)
        
