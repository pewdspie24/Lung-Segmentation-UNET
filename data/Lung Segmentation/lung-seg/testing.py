import cv2
import glob
import os
import numpy as np

img_path = './testLung_img'
mask_path = './testLung_mask'
img_after_path = './eval'
images = os.listdir(img_path)

for filename in images:
    img = cv2.imread(os.path.join(img_path,filename))
    mask = cv2.imread(os.path.join(mask_path,filename))
    rows,cols = 320, 320
    for i in range(rows):
        for j in range(cols):
            k = mask[i,j]
            if k[0] >= 10:
                img[i][j] = 0,255,0
    cv2.imwrite(os.path.join(img_after_path, filename), img)
