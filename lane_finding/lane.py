# Define a class to receive the characteristics of each line detection
import numpy as np
import math
from collections import deque

from lane_finding.fit_lines import xm_per_pix

class Line():
    def __init__(self):
        self.n = 5
        # was the line detected in the last iteration?
        self.detected = False
        self.all_left_fits = deque()
        self.all_right_fits = deque()
        self.all_cl = deque()
        self.all_cr = deque()
        # x values of the last n fits of the line
        self.recent_xfitted = deque([])
        #polynomial coefficients averaged over the last n iterations
        self.best_fit_left = None
        self.best_fit_right = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #y values for detected line pixels
        self.leftx = None
        self.rightx = None

    def add(self, left_fit, right_fit, cl, cr):
        self.all_left_fits.append(left_fit)
        self.all_right_fits.append(right_fit)
        self.all_cl.append(cl)
        self.all_cr.append(cr)
        if len(self.all_left_fits) > self.n:
            self.all_left_fits.popleft()
            self.all_right_fits.popleft()
            self.all_cl.popleft()
            self.all_cr.popleft()

    def average(self):
        self.best_fit_left = np.sum(self.all_left_fits, axis=0) / len(self.all_left_fits)
        self.best_fit_right = np.sum(self.all_right_fits, axis=0) / len(self.all_right_fits)
        self.leftx = self.intersect(self.best_fit_left)
        self.rightx = self.intersect(self.best_fit_right)
        cl = np.sum(self.all_cl)/len(self.all_cl)
        cr = np.sum(self.all_cr)/len(self.all_cr)
        self.radius_of_curvature = (cl + cr) / 2
        self.line_base_pos = self.compute_offset()

    def intersect(self, fit):
        return fit[0]*720**2 + fit[1]*720 + fit[2]

    def compute_offset(self):
        lx = self.intersect(self.best_fit_left)
        rx = self.intersect(self.best_fit_right)
        center = lx + (rx-lx)/2
        center_m = (640-center) * xm_per_pix
        return center_m

    def sane(self, left_fit, right_fit):
        if self.leftx is None: # accept the first measurment
            return True

        max_diff = 10
        leftx = self.intersect(left_fit)
        rightx = self.intersect(right_fit)
        dl = math.fabs(leftx-self.leftx)
        dr = math.fabs(rightx-self.rightx)
        if dl < max_diff and dr < max_diff:
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

