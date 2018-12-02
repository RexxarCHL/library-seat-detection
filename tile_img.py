import numpy as np
import cv2

img = []
for i in range(4):
    img.append(cv2.imread("img/img{}.jpg".format(i), cv2.IMREAD_COLOR))

img = np.array(img)

tiled = np.hstack(img)
cv2.imwrite("img/img_tiled.jpg", tiled)

stack = np.zeros((2, 1080, 1920*2, 3))
stack[0] = np.hstack(img[:2])
stack[1] = np.hstack(img[2:])
stack = np.vstack(stack)
cv2.imwrite("img/img_stacked.jpg", stack)
