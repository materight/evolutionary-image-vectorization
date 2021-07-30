import cv2 as cv
import numpy as np
from PIL import Image

img = cv.cvtColor(np.array(Image.open(f'samples/starry_night.jpg')), cv.COLOR_RGB2BGR)
img = cv.resize(img, None, fx=.5, fy=.5)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_filtered = cv.bilateralFilter(gray, 9, 80, 50)

sigma = 0.33 # 0.33
v = np.median(gray_filtered)
# apply automatic Canny edge detection using the computed median
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
# Extract image contours
target = cv.Canny(gray_filtered, lower, upper)
print(lower, ',', upper)

cv.imshow('original', img)
cv.imshow('gray_filtered1', cv.bilateralFilter(gray, 9, 80, 50))
cv.imshow('contours1', target)

cv.waitKey(0)
cv.destroyAllWindows()