import pickle
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Define a class to receive the characteristics of each line detection
class TrackLine():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # counter
        self.counter = 0

        # x values of the last n fits of the left line
        self.recent_xfitted_left = [np.array([0,0,0], dtype='float')]

        # x values of the last n fits of the right line
        self.recent_xfitted_right = [np.array([0,0,0], dtype='float')]

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent left fit
        self.current_left_fit = [np.array([False])]

        # polynomial coefficients for the most recent right fit
        self.current_right_fit = [np.array([False])]

        # radius of curvature of the left line in m
        self.left_radius_of_curvature = None

        # radius of curvature of the right line in m
        self.right_radius_of_curvature = None

        # radius of curvation in m
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        # all detected left line pixels
        self.all_left = None

        # all detected right line pixels
        self.all_right = None

        # Set the width of the windows +/- margin
        self.win_margin = 60

        # Set minimum number of pixels found to recenter window
        self.win_minpix = 50

    def find_lane(self, binary_warped):

        if not self.detected:
            out_img = self.search_new_lanes(binary_warped)
        else:
            out_img = self.update_lanes(binary_warped)

        return out_img

    def search_new_lanes(self, binary_warped):

        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[np.ceil(binary_warped.shape[0]/2).astype(int):, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        quarter = np.int(histogram.shape[0]/4)
        three_quarter = np.int(histogram.shape[0]*3/4)
        leftx_base = np.argmax(histogram[quarter:midpoint]) + quarter
        rightx_base = np.argmax(histogram[midpoint:three_quarter]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # set minimum required number of lane pixels
        min_pixels = self.win_minpix*(nwindows-1)

        # identify lane pixels based on histogram information
        leftx, lefty = \
            self.identify_lane_pixels(binary_warped, nwindows, leftx_base, nonzerox, nonzeroy)

        rightx, righty = \
            self.identify_lane_pixels(binary_warped, nwindows, rightx_base, nonzerox, nonzeroy)

        if len(leftx) > min_pixels or len(rightx) > min_pixels:
            self.detected = True

            self.all_left = [leftx, lefty]
            self.all_right = [rightx, righty]

            # Fit a second order polynomial to each
            self.current_left_fit = np.polyfit(lefty, leftx, 2)
            self.current_right_fit = np.polyfit(righty, rightx, 2)

            res_img = self.draw_warped(binary_warped)

        else:
            self.detected = False

        return res_img

    def update_lanes(self, binary_warped):

        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # set minimum required number of lane pixels
        min_pixels = self.win_minpix * 8

        left_lane_inds = (
            (nonzerox > (self.current_left_fit[0] * (nonzeroy ** 2) + self.current_left_fit[1] * nonzeroy +
                         self.current_left_fit[2] - self.win_margin)) & (
                nonzerox < (self.current_left_fit[0] * (nonzeroy ** 2) + self.current_left_fit[1] * nonzeroy +
                            self.current_left_fit[2] + self.win_margin)))
        right_lane_inds = (
            (nonzerox > (self.current_right_fit[0] * (nonzeroy ** 2) + self.current_right_fit[1] * nonzeroy +
                         self.current_right_fit[2] - self.win_margin)) & (
                nonzerox < (self.current_right_fit[0] * (nonzeroy ** 2) + self.current_right_fit[1] * nonzeroy +
                            self.current_right_fit[2] + self.win_margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # not enough
        if len(leftx) > min_pixels and len(rightx) > min_pixels:
            self.detected = True

            self.all_left = [leftx, lefty]
            self.all_right = [rightx, righty]

            # Fit a second order polynomial to each
            self.current_left_fit = np.polyfit(lefty, leftx, 2)
            self.current_right_fit = np.polyfit(righty, rightx, 2)

            res_img = self.draw_warped(binary_warped)
            self.counter += 1
            if self.counter > 5:
                self.counter = 0
                self.detected = False

        else:
            self.detected = False
            res_img = self.search_new_lanes(binary_warped)

        return res_img

    def identify_lane_pixels(self, binary_warped, nwindows, x_base, nonzerox, nonzeroy):

        # set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)

        # create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # current positions to be updated for each window
        x_current = x_base

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_x_low = x_current - self.win_margin
            win_x_high = x_current + self.win_margin

            # identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
                nonzerox < win_x_high)).nonzero()[0]
            lane_inds.append(good_inds)

            # if minpix pixels found, recenter next window on their mean position
            if len(good_inds) > self.win_minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        return x, y

    def draw_warped(self, binary_warped):

        # create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = self.current_left_fit[0] * ploty ** 2 + self.current_left_fit[1] * ploty + self.current_left_fit[2]
        right_fitx = self.current_right_fit[0] * ploty ** 2 + self.current_right_fit[1] * ploty + self.current_right_fit[2]

        left_fitx = np.array([min(item, binary_warped.shape[1]-20) for item in left_fitx])
        right_fitx = np.array([min(item, binary_warped.shape[1]-20) for item in right_fitx])
        left_fitx = np.array([max(item, 20) for item in left_fitx])
        right_fitx = np.array([max(item, 20) for item in right_fitx])

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self.win_margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self.win_margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self.win_margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self.win_margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

        out_img[self.all_left[1], self.all_left[0]] = [255, 0, 0]
        out_img[self.all_right[1], self.all_right[0]] = [0, 0, 255]

        out_img[ploty.astype(int), left_fitx.astype(int)] = [0, 255, 255]
        out_img[ploty.astype(int), right_fitx.astype(int)] = [0, 255, 255]

        res_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return res_img

    def draw_on_original(self, binary_warped, Minv, undist):

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        line_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = self.current_left_fit[0] * ploty ** 2 + self.current_left_fit[1] * ploty + self.current_left_fit[2]
        right_fitx = self.current_right_fit[0] * ploty ** 2 + self.current_right_fit[1] * ploty + \
                     self.current_right_fit[2]

        left_fitx = np.array([min(item, binary_warped.shape[1] - 20) for item in left_fitx])
        right_fitx = np.array([min(item, binary_warped.shape[1] - 20) for item in right_fitx])
        left_fitx = np.array([max(item, 20) for item in left_fitx])
        right_fitx = np.array([max(item, 20) for item in right_fitx])

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        line_thickness = 10
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - line_thickness, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + line_thickness, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(line_warp, np.int_([left_line_pts]), (255, 0, 0))
        cv2.fillPoly(line_warp, np.int_([right_line_pts]), (0, 0, 255))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
        new_line_warp = cv2.warpPerspective(line_warp, Minv, (undist.shape[1], undist.shape[0]))
        # Combine the result with the original image
        res_img = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        res_img = cv2.addWeighted(res_img, 1, new_line_warp, 0.6, 0)

        return res_img

    def calculate_curvature(self, binary_warped):

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        y_eval = np.max(ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 600  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.all_left[1] * ym_per_pix, self.all_left[0] * xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.all_right[1] * ym_per_pix, self.all_right[0] * xm_per_pix, 2)

        self.line_base_pos = xm_per_pix*binary_warped.shape[1]/2- 0.5*(left_fit_cr[2] + right_fit_cr[2])

        # Calculate the new radii of curvature
        self.left_radius_of_curvature = \
            ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        self.right_radius_of_curvature = \
            ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        self.radius_of_curvature = np.mean([self.left_radius_of_curvature, self.right_radius_of_curvature])

        # Now our radius of curvature is in meters
        print('left: ', self.left_radius_of_curvature, 'm; right: ', self.right_radius_of_curvature, 'm; mean: ',
              self.radius_of_curvature)
