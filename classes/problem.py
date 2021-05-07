import numpy as np
import cv2 as cv

class Problem:
    EA = 1
    PSO = 2
    
    def __init__(self, problem_type, target, internal_resolution):
        self.problem_type = problem_type
        self.scale_factor = internal_resolution / min(target.shape[:2]) if internal_resolution > 0 else 1
        self.set_target(target)

    def set_target(self, target):
        if self.problem_type == self.EA:
            self.target = cv.resize(target, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        elif self.problem_type == self.PSO:
            gray_filtered = cv.bilateralFilter(target, 7, 50, 50)
            self.target = cv.Canny(gray_filtered, 120, 140)
            self.target = np.where(self.target > 0, 0, 255).astype(np.uint8)
            self.target = cv.distanceTransform(self.target, cv.DIST_C, 3)
            self.target = cv.normalize(self.target, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        
