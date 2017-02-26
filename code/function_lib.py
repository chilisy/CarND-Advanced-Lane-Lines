import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import global_setting


def undistort_image(img, calib_data_file):

    with open(calib_data_file, 'rb') as f:
        dist_pickle = pickle.load(f)
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]

    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist


def dir_threshold(channel, sobel_kernel=3, thresh=(0, np.pi / 2)):

    # 2) Take the gradient in x and y separately
    sob_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sob_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    sob_x_abs = np.abs(sob_x)
    sob_y_abs = np.abs(sob_y)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    abs_grad_dir = np.arctan2(sob_y_abs, sob_x_abs)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(abs_grad_dir)
    binary_output[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def mag_thresh(channel, sobel_kernel=3, thresh=(0, 255)):

    # 2) Take the gradient in x and y separately
    sob_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sob_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    magnitude_gradient = np.sqrt(sob_x**2 + sob_y**2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(magnitude_gradient)/255
    magnitude_gradient = (magnitude_gradient/scale_factor).astype(np.uint8)

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(magnitude_gradient)
    binary_output[(magnitude_gradient >= thresh[0]) & (magnitude_gradient <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def abs_sobel_thresh(channel, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def hls_select(img, coord='s', thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # 2) Apply a threshold to the S channel
    if coord == 's':
        channel = hls[:, :, 2]
    elif coord == 'l':
        channel = hls[:, :, 1]
    else:
        channel = hls[:, :, 0]

    binary_output = np.zeros_like(channel)

    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1

    # 3) Return a binary image of threshold result
    return channel, binary_output


def combine_sobel(channel, abs_thres, dir_thres, kernel_size=3):

    gradx = abs_sobel_thresh(channel, orient='x', sobel_kernel=kernel_size, thresh=abs_thres)
    grady = abs_sobel_thresh(channel, orient='y', sobel_kernel=kernel_size, thresh=abs_thres)
    mag_binary = mag_thresh(channel, sobel_kernel=kernel_size, thresh=abs_thres)
    dir_binary = dir_threshold(channel, sobel_kernel=kernel_size, thresh=dir_thres)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) | (dir_binary == 1))] = 1

    return combined


def warp_image(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[img_size[0]*0.425, img_size[1] * 0.65],
         [img_size[0]*0.04, img_size[1]],
         [img_size[0]*0.96, img_size[1]],
         [img_size[0]*0.575, img_size[1] * 0.65]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(src, dst)

    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, Minv


def process_image(img, track_line):
    # parameters
    calib_data_file = 'calib_data.p'

    # sobel kernel size
    ksize = 3

    # thresholds
    abs_thres = (40, 80)
    dir_thres = (0.01, 0.055)

    # step 1: undistort image
    undist = undistort_image(img, calib_data_file)

    # step 2: warp image
    warped_img, Minv = warp_image(undist)

    # step 3: convert to hls channel
    s_channel, s_hls_binary = hls_select(warped_img, coord='s', thresh=(0, 255))
    l_channel, l_hls_binary = hls_select(warped_img, coord='l', thresh=(200, 255))
    #h_channel, hls_binary = hls_select(img, coord='h', thresh=(0, 255))

    # step 4: apply sobel
    s_sobel_binary = combine_sobel(s_channel, abs_thres, dir_thres, kernel_size=ksize)
    l_sobel_binary = combine_sobel(l_channel, abs_thres, dir_thres, kernel_size=ksize)
    sobel_binary = np.zeros_like(s_sobel_binary)
    sobel_binary[(s_sobel_binary == 1) | (l_sobel_binary == 1)] = 1

    # step 5: detect line
    out_img_warped = track_line.find_lane(sobel_binary)

    # step 6: calculate curvature
    track_line.calculate_curvature(sobel_binary)

    # step 7: unwarp image and draw
    out_img = track_line.draw_on_original(sobel_binary, Minv, undist)

    return undist, warped_img, sobel_binary, s_channel, l_channel, out_img_warped, out_img


def process_image_for_video(img):

    undist, warped_img, sobel_binary, s_channel, l_channel, out_img_warped, out_img = process_image(img, global_setting.trackline)

    sobel_binary = sobel_binary.astype(np.uint8)
    sobel_binary = sobel_binary*255
    sobel_binary = cv2.cvtColor(sobel_binary, cv2.COLOR_GRAY2RGB)

    s_channel = cv2.cvtColor(s_channel, cv2.COLOR_GRAY2RGB)
    l_channel = cv2.cvtColor(l_channel, cv2.COLOR_GRAY2RGB)

    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    diagScreen[0:720, 0:1280] = out_img
    diagScreen[0:360, 1280:1920] = cv2.resize(out_img_warped, (640, 360), interpolation=cv2.INTER_AREA)
    diagScreen[360:720, 1280:1920] = cv2.resize(warped_img, (640, 360), interpolation=cv2.INTER_AREA)
    diagScreen[720:1080, 1280:1920] = cv2.resize(sobel_binary, (640, 360), interpolation=cv2.INTER_AREA)
    diagScreen[720:1080, 0:640] = cv2.resize(s_channel, (640, 360), interpolation=cv2.INTER_AREA)
    diagScreen[720:1080, 640:1280] = cv2.resize(l_channel, (640, 360), interpolation=cv2.INTER_AREA)

    # curvature text
    curvature_text = 'est. radius of curvature: ' + str(global_setting.trackline.radius_of_curvature) + ' m'
    pos_text = 'deviation from middle: ' + str(global_setting.trackline.line_base_pos) + ' m'

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(diagScreen, curvature_text, (80, 1000), font, 1, (255, 255, 255), 2)
    cv2.putText(diagScreen, pos_text, (80, 1040), font, 1, (255, 255, 255), 2)

    return diagScreen
