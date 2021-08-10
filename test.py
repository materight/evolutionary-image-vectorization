import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw

LENGTH = 40
scale = 5

# Compute
center, rotation = np.array([50, 50]), 0

while True:
    # Compute
    rotation += .01
    r = LENGTH / 2
    d = [r * np.cos(rotation), r * np.sin(rotation)]
    cd = [np.cos(rotation+np.pi/2), np.sin(rotation+np.pi/2)]
    centerL = center + cd
    centerR = center - cd
    
    coords = np.concatenate([center + d, center - d]) 
    coordsL = np.concatenate([centerL + d, centerL - d]) 
    coordsR = np.concatenate([centerR + d, centerR - d]) 
    

    # Draw
    img = Image.new('RGB', (100*scale, 100*scale), color='black')
    draw = ImageDraw.Draw(img, 'RGB')
    draw.line(tuple(coords*scale), fill=(255,255,255), width=2)
    draw.line(tuple(coordsL*scale), fill=(0,255,0), width=2)
    draw.line(tuple(coordsR*scale), fill=(255,0,0), width=2)
    img = np.array(img)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    # Show
    cv.imshow('Img', img)
    cv.waitKey(1)