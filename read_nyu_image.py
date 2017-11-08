import cv2
import h5py
import numpy as np

filepath = 'nyu_depth_v2_labeled.mat'
arrays = {}
f = h5py.File(filepath)
# print(f.keys())
for k, v in f.items():
    arrays[k] = v

images = arrays['images']
first_pic = np.transpose(images[0], [2, 1, 0])  # images[0].shape -> Nx3xWxH
first_pic = first_pic[:, :, [2, 1, 0]]  # first_pic是bgr 需要转成rgb
cv2.imwrite('test.png', first_pic)

depths = arrays['depths']
d = depths[0]
d = (d - 1.5) / 2.5 * 255  # 因为这个图的范围在1.7-3.6 hardcode一下
first_depth = np.transpose(d, [1, 0]).astype('uint8')
first_depth = cv2.applyColorMap(first_depth, cv2.COLORMAP_JET)
cv2.imwrite("./nyu_depth_test.jpg", first_depth)
