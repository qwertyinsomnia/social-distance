import h5py
import numpy as np
import cv2

input_h5_file_path = "scene_002.h5"
input_h5 = h5py.File(input_h5_file_path, 'r')
index = 10

input_images_data = input_h5['image']
in_height, in_width = input_images_data.shape[2:]
print(input_images_data.shape)

image = np.array(input_images_data[index])
image = image.transpose(1, 2, 0)
print(image.shape)

cv2.imshow("win", image)
cv2.waitKey(0)


src = np.float32([[0, 96], [512, 96], [0, 384], [512, 384]])
dst = np.float32([[0, 0], [700, 0], [200, 384], [500, 384]])

m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(image, m, (700, 384))
cv2.imshow("win", result)
cv2.waitKey(0)

camera_angles = input_h5['camera_angle']
print(list(camera_angles)[index] * 180)
camera_heights = input_h5['camera_height']
print(list(camera_heights)[index])
# camera_fus = input_h5['camera_fu']
# print(list(camera_fus))
# camera_fvs = input_h5['camera_fv']
# print(list(camera_fvs))

input_h5.close()