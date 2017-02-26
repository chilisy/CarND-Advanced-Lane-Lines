import pickle
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from function_lib import *
import global_setting

# test images
img_dir = '../test_images/'
output_dir = '../output_images/'

# get image files
all_files = os.listdir(img_dir)
img_files = [file_names for file_names in all_files if not file_names[0] == '.']

# undistort images
save_dir = output_dir + 'final/res_'

#img_files = ['test5.jpg']
global_setting.init()

for img_file in img_files:
    img = cv2.imread(img_dir + img_file)

    out_img = process_image_for_video(img)
    global_setting.trackline.detected = False

    #sobel_output = sobel_output.astype(np.uint8)
    #sobel_output = sobel_output*255
    #sobel_output = cv2.cvtColor(sobel_output, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(save_dir + img_file, out_img)

