import numpy as np
import cv2


# depth_ndarray = np.ones((256,256), dtype='int8').reshape(256, 256) * 10
# depth_ndarray = np.dstack((depth_ndarray, depth_ndarray, depth_ndarray))
# print(depth_ndarray)
# print('-------------------------')
# print(pic.shape)
# print(pic[150,150])
im_gray = cv2.imread("moon.png",cv2.IMREAD_GRAYSCALE)
print(im_gray.shape)

depth_ndarray = (np.arange(256 * 256) % 256).reshape(256, 256).astype('uint8')# 注意最后这个astype很重要 不然类型不满足要求

depth_ndarray_3d = np.dstack((depth_ndarray, depth_ndarray, depth_ndarray))
cv2.imwrite("./array.jpg", depth_ndarray_3d)

# im_gray = cv2.imread("pluto.jpg", cv2.IMREAD_GRAYSCALE)
print(depth_ndarray.shape)
im_color = cv2.applyColorMap(depth_ndarray, cv2.COLORMAP_JET)
cv2.imwrite("./array_rgb.jpg", im_color)

# cv2.imshow('moon', depth_ndarray)
# cv2.waitKey(0)