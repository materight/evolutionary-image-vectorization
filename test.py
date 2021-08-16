import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw

v=cv.VideoCapture('samples/cars.gif')
print(v.get(cv.CAP_PROP_FPS))