import cv2
import glob
import os

img_path = './test'
# print(os.listdir(img_path))
img_after_path = './testLung_img'
images = glob.glob(img_path+"/*.png")

for path in images:
    filename = path.split('/')[-1]
    img = cv2.imread(path)
    img = cv2.resize(img, (320,320))
    cv2.imwrite(os.path.join(img_after_path, filename), img)

# img = cv2.imread(images[0])
# rows,cols = img.shape[1], img.shape[2]

# for i in range(rows):
#     for j in range(cols):
#         k = img[i,j]
#         print(k)
