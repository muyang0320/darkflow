import cv2
import numpy as np
filepath = 'nyu_depth_v2_labeled.mat'
arrays = {}
f = h5py.File(filepath)
# print(f.keys())
for k,v in f.items():
    arrays[k] = v

images = arrays['images']
first_pic = np.transpose(images[0], [2,1,0]) # images[0].shape -> Nx3xWxH
first_pic = first_pic[:,:,[2,1,0]] # first_pic是bgr 需要转成rgb
# cv2.imwrite('test.png',first_pic)