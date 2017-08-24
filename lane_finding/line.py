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

    def _add(self, left_fit, right_fit, cl, cr):
        """Add the current measurement to the lists used for averaging."""
        self.all_left_fits.append(left_fit)
        self.all_right_fits.append(right_fit)
        self.all_left_curvatures.append(cl)
        self.all_right_curvatures.append(cr)
        if len(self.all_left_fits) > self.n:
            self.all_left_fits.popleft()
            self.all_right_fits.popleft()
            self.all_left_curvatures.popleft()
            self.all_right_curvatures.popleft()

    def _average(self):
        """Update all averages from the lists of most recent measurements."""
        self.best_fit_left = np.sum(self.all_left_fits, axis=0) / len(self.all_left_fits)
        self.best_fit_right = np.sum(self.all_right_fits, axis=0) / len(self.all_right_fits)
        self.leftx = self._intersect(self.best_fit_left)
        self.rightx = self._intersect(self.best_fit_right)
        cl = np.sum(self.all_left_curvatures)/len(self.all_left_curvatures)
        cr = np.sum(self.all_right_curvatures)/len(self.all_right_curvatures)
        self.radius_of_curvature = (cl + cr) / 2
        self.line_base_pos = self._compute_offset()

    def _check_measurement(self, left_fit, right_fit):
        """Check whether the current measurement fits with the tracked average."""
        # always accept the first measurement
        if self.leftx is None:
            return True

        # reject line fits that differ too much from the average start of the
        # lane at the bottom of the image
        max_diff_pixels = 10
        leftx = self._intersect(left_fit)
        rightx = self._intersect(right_fit)
        dl = math.fabs(leftx-self.leftx)
        dr = math.fabs(rightx-self.rightx)
        if dl < max_diff_pixels and dr < max_diff_pixels:
            return True
        else:
            print("diff left: {:.2f}, right: {:.2f}".format(dl, dr))
        return False

    def _compute_offset(self):
        """Calculate the offset of the center of the lane from the ideal."""
        center = self.leftx + (self.rightx-self.leftx)/2
        center_m = (ideal_image_center-center) * xm_per_pix
        return center_m

    def _intersect(self, fit):
        """Calculate the intersection of a fit with the bottom of the image."""
        return fit[0]*image_height**2 + fit[1]*image_height + fit[2]

    def update(self, left_fit, right_fit, cl, cr, force=False):
        """Update the line tracker with the current estimations of fit and curvature."""
        if self._check_measurement(left_fit, right_fit) or force:
            self._add(left_fit, right_fit, cl, cr)
            self._average()
            self.detected = True
        else:
            print("Rejecting current measurement.")
            self.detected = False
        return self.detected

