# Define a class to receive the characteristics of each line detection
import numpy as np
import math
from collections import deque

from lane_finding.fit_lines import xm_per_pix, ideal_image_center, image_height

class Line():
    def __init__(self):
        # length of averaging window
        self.n = 5
        # was the line detected in the last iteration?
        self.detected = False
        # list of left polynomials used for averaging
        self.all_left_fits = deque()
        # list of right polynomials used for averaging
        self.all_right_fits = deque()
        # list of left curvatures used for averaging
        self.all_left_curvatures = deque()
        # list of right curvatures used for averaging
        self.all_right_curvatures = deque()
        # left polynomial coefficients averaged over the last n iterations
        self.best_fit_left = None
        # right polynomial coefficients averaged over the last n iterations
        self.best_fit_right = None
        # radius of curvature of the line in meters
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # intersection of the left lane with the bottom of the image
        self.leftx = None
        # intersection of the right lane with the bottom of the image
        self.rightx = None

    def add(self, left_fit, right_fit, cl, cr):
        self.all_left_fits.append(left_fit)
        self.all_right_fits.append(right_fit)
        self.all_left_curvatures.append(cl)
        self.all_right_curvatures.append(cr)
        if len(self.all_left_fits) > self.n:
            self.all_left_fits.popleft()
            self.all_right_fits.popleft()
            self.all_left_curvatures.popleft()
            self.all_right_curvatures.popleft()

    def average(self):
        self.best_fit_left = np.sum(self.all_left_fits, axis=0) / len(self.all_left_fits)
        self.best_fit_right = np.sum(self.all_right_fits, axis=0) / len(self.all_right_fits)
        self.leftx = self.intersect(self.best_fit_left)
        self.rightx = self.intersect(self.best_fit_right)
        cl = np.sum(self.all_left_curvatures)/len(self.all_left_curvatures)
        cr = np.sum(self.all_right_curvatures)/len(self.all_right_curvatures)
        self.radius_of_curvature = (cl + cr) / 2
        self.line_base_pos = self.compute_offset()

    def intersect(self, fit):
        return fit[0]*image_height**2 + fit[1]*image_height + fit[2]

    def compute_offset(self):
        center = self.leftx + (self.rightx-self.leftx)/2
        center_m = (ideal_image_center-center) * xm_per_pix
        return center_m

    def sane(self, left_fit, right_fit):
        if self.leftx is None: # accept the first measurment
            return True

        max_diff_pixels = 10
        leftx = self.intersect(left_fit)
        rightx = self.intersect(right_fit)
        dl = math.fabs(leftx-self.leftx)
        dr = math.fabs(rightx-self.rightx)
        if dl < max_diff_pixels and dr < max_diff_pixels:
            return True
        else:
            print("diff left: {:.2f}, right: {:.2f}".format(dl, dr))
        return False

    def update(self, left_fit, right_fit, cl, cr, force=False):
        if self.sane(left_fit, right_fit) or force:
            self.add(left_fit, right_fit, cl, cr)
            self.average()
            self.detected = True
        else:
            print("Ignoring insane measurement.")
            self.detected = False
        return self.detected

