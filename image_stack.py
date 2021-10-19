import os
import cv2
import numpy as np

files = []

Result = []

for file in os.listdir('./error_images'):
    files.append(file)

print(files)

for i in range(10):
    cols = []
    for j in range(10):
        if i != j and f'output_{i}_label_{j}.jpg' in files:
            img = cv2.imread(f'error_images/output_{i}_label_{j}.jpg')
        else:
            img = np.zeros((32, 32, 3)) * 1.0
        cols.append(img)  
    np.stack(cols)
    Result.append(np.hstack(cols))

Result = np.vstack(Result)
cv2.imwrite('./Result.jpg', Result)


