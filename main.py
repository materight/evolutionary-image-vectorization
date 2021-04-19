import cv2 as cv

img = cv.imread('img/starry-night.jpg')

cv.imshow('img', img)
cv.waitKey(0)