import numpy as np
import cv2 as cv

class Problem:
    def __init__(self, target, internal_resolution):
        self.scale_factor = internal_resolution / min(target.shape[:2]) if internal_resolution > 0 else 1
        self.set_target(target)

    def set_target(self, target):
        self.target = cv.resize(target, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
