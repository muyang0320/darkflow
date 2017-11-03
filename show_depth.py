import numpy as np
import cv2

pic = cv2.imread('./moon.png')

# depth_ndarray = np.ones((256,256), dtype='int8').reshape(256, 256) * 10
# depth_ndarray = np.dstack((depth_ndarray, depth_ndarray, depth_ndarray))
# print(depth_ndarray)
# print('-------------------------')
# print(pic.shape)
# print(pic[150,150])

depth_ndarray = (np.arange(256 * 256, dtype='int8') % 256).reshape(256, 256)
depth_ndarray = np.dstack((depth_ndarray, depth_ndarray, depth_ndarray))
cv2.imwrite("./array.jpg", depth_ndarray)
# cv2.imshow('moon', depth_ndarray)
# cv2.waitKey(0)