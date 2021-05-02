from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time


from classes.ea import EA

# Set windows properties
cv.namedWindow('Result')
# cv.setWindowProperty('Result', cv.WND_PROP_TOPMOST, 1)

# Load image
img = Image.open('img/mona-lisa.jpg')
img = np.array(img)
img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
img = cv.resize(img, (0, 0), fx=.5, fy=.5)
print(f'Image size: {img.shape}')

cap = cv.VideoCapture(0)

while True:
    res, img = cap.read()
    img = cv.resize(img, (0, 0), fx=.5, fy=.5)
    gray_filtered = cv.bilateralFilter(img, 7, 50, 50)
    img = cv.Canny(gray_filtered, 120, 140)
    img = np.where(img > 0, 0, 255).astype(np.uint8)

    field = cv.distanceTransform(img.copy(), cv.DIST_C, 3)
    field = cv.normalize(field, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    #img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    cv.imshow('Result', np.hstack([img, field]))
    cv.waitKey(1)

plt.imshow(field, cmap='inferno')
plt.show()

'''
# Genetic algorithm
ea = EA(
    img,
    pop_size=50,
    n_poly=50,
    n_vertex=3,
    selection_cutoff=.1,
    mutation_chances=(0.01, 0.01, 0.01),
    mutation_factors=(0.2, 0.2, 0.2)
)

hbest, havg, hworst = [], [], []
while True:
    start_time = time.time()
    gen, best, population = ea.next()
    
    print(f'{gen}) {round((time.time() - start_time)*1000)}ms, best: {best.fitness}, ({best.n_poly} poly)')
    hbest.append(best.fitness)
    havg.append(np.average([ind.fitness for ind in population]))
    hworst.append(population[-1].fitness)
    
    best_img = cv.resize(best.draw(), img.shape[1::-1])
    cv.imshow('Result', np.hstack([img, best_img]))
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break 

cv.destroyAllWindows()
x = range(len(hbest))
plt.plot(x, hbest, c='r', label='best')
#plt.plot(x, havg, c='g', label='avg')
#plt.plot(x, hworst, c='k', label='worst')
plt.legend()
plt.show()
'''