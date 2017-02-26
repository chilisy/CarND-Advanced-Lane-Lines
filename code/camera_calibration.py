import pickle
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


dir = "../camera_cal/"
p_file = "calib_data.p"

# get all image files in the folder
all_files = os.listdir(dir)
img_files = [file_names for file_names in all_files if not file_names[0] == '.']

nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.


for idx, fname in enumerate(img_files):
    img = cv2.imread(dir + fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

test_image = cv2.imread(dir + img_files[1])
img_size = (test_image.shape[1], test_image.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

calib_data = {'mtx': mtx, 'dist': dist}

with open(p_file, 'wb') as f:
    pickle.dump(calib_data, f)

# undistort test image and show it
undist = cv2.undistort(test_image, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undist)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.show()

