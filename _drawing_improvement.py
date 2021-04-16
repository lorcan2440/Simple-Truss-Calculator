import cv2
import numpy as np

original_image = cv2.imread('JointEOriginal.jpg')

width, height = 2970, 2100
points_1 = np.float32([[384, 213], [3843, 303], [408, 2713], [3808, 2665]])
points_2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

matrix = cv2.getPerspectiveTransform(points_1, points_2)
image_out = cv2.warpPerspective(original_image, matrix, (width, height))

cv2.imwrite('JointEClean.png', image_out)
