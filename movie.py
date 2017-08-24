import os
import glob
import matplotlib.image as mplimg
import numpy as np
import cv2
from lane_finding.undistort import undistort, warp_to_lane
from lane_finding.threshold import threshold_basic
from lane_finding.fit_lines import fit_lanes, track_lanes
from lane_finding.fit_lines import plot_windows, plot_lanes, augment_image_with_lane, write_text, track_lanes
from lane_finding.lane import Line

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
line = Line()

def process_image(img):
    global line
    undist = undistort(img)
    warped = warp_to_lane(undist)
    binary_warped = threshold_basic(warped)
    try:
        if not line.detected:
            left_fit, right_fit, left_lane_inds, right_lane_inds, cl, cr, windows = fit_lanes(binary_warped)
            line.update(left_fit, right_fit, cl, cr, force=True)
        else:
            left_fit, right_fit, left_lane_inds, right_lane_inds, cl, cr, windows = \
            track_lanes(binary_warped, line.best_fit_left, line.best_fit_right)
            line.update(left_fit, right_fit, cl, cr)
        curves = plot_lanes(binary_warped, left_fit, right_fit, left_lane_inds, right_lane_inds)
        curves = plot_windows(curves, windows)
        lanes = augment_image_with_lane(undist, line.best_fit_left, line.best_fit_right)
        lanes = write_text(lanes, line.radius_of_curvature, line.line_base_pos)
    except TypeError as e:
        print(e)
        lanes = undist
    return lanes

#white_output = 'project_video_augmented.mp4'
#filename = "challenge_video.mp4"
filename = "project_video.mp4"

name, ext = filename.split('.')
result_filename = name + "_augmented." + ext
print("input: {}, output: {}".format(filename, result_filename))

clip1 = VideoFileClip(filename)#.subclip(35, 45)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color

white_clip.write_videofile(result_filename, audio=False)
