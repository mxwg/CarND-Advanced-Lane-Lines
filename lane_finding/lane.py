# Define a class to receive the characteristics of each line detection
import numpy as np
import math
from collections import deque

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
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit_left = None
        self.best_fit_right = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
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
        self.leftx = self.best_fit_left[2]
        self.rightx = self.best_fit_right[2]
        cl = np.sum(self.all_cl)/len(self.all_cl)
        cr = np.sum(self.all_cr)/len(self.all_cr)
        self.radius_of_curvature = (cl + cr) / 2

    def sane(self, left_fit, right_fit):
        if self.leftx is None: # accept the first measurment
            return True
        max_diff = 15
        leftx = left_fit[2]
        rightx = right_fit[2]
        dl = math.fabs(leftx-self.leftx)
        dr = math.fabs(rightx-self.rightx)
        if dl < 10 and dr < 10:
            return True

    def update(self, left_fit, right_fit, cl, cr, force=False):
        if self.sane(left_fit, right_fit) or force:
            self.add(left_fit, right_fit, cl, cr)
            self.average()
            self.detected = True
        else:
            print("Ignoring insane measurement.")
            self.detected = False
        return self.detected

