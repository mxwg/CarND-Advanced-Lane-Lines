# Run the whole pipeline on all images of a folder
# and save the results to "output_images"
import os
import glob
import matplotlib.image as mplimg
import numpy as np
import cv2
import math
import shutil
from lane_finding.undistort import undistort, warp_to_lane
from lane_finding.threshold import threshold_basic
from lane_finding.fit_lines import fit_lanes, track_lanes, plot_windows, \
        plot_lanes, plot_lanes_only, augment_image_with_lane
from lane_finding.fit_lines import write_curvature_and_offset, info

# Get images
input_folder = "test_images/track3"
input_folder = "images_project_video"
output_folder = "output_images"
try:
    shutil.rmtree(output_folder)
except FileNotFoundError:
    pass
os.mkdir(output_folder)

def save(prefix, image_name, image, cmap=None):
    file_name = os.path.join(output_folder, prefix + "_" + os.path.basename(image_name))
    mplimg.imsave(file_name, image, cmap=cmap)

image_names = glob.glob(os.path.join(input_folder, "output*.jpg"))

np.set_printoptions(precision=6, suppress=True)
from lane_finding.line import Line
line = Line()
for image in image_names[0:10]:
    print("working on image {}".format(image))
    try:
        img = mplimg.imread(image)
        undist = undistort(img)
        warped = warp_to_lane(undist)
        binary_warped = threshold_basic(warped)
        #save("binary_warped", image, binary_warped, cmap='gray')
        if not line.detected:
            print("Full detection...")
            left_fit, right_fit, left_lane_inds, right_lane_inds, cl, cr, windows = fit_lanes(binary_warped)
            line.update(left_fit, right_fit, cl, cr, force=True)
        else:
            left_fit, right_fit, left_lane_inds, right_lane_inds, cl, cr, windows = \
            track_lanes(binary_warped, line.best_fit_left, line.best_fit_right)
            line.update(left_fit, right_fit, cl, cr)
        curves = plot_lanes(binary_warped, left_fit, right_fit, left_lane_inds,
                            right_lane_inds, line.detected)
        curves = plot_windows(curves, windows)
        curves = info(curves, "detected" if line.detected else "not detected")
        if line.best_fit_left is not None:
            curves = plot_lanes_only(curves, line.best_fit_left, line.best_fit_right)
        save("curves", image, curves)
        lanes = augment_image_with_lane(undist, line.best_fit_left, line.best_fit_right)
        lanes = write_curvature_and_offset(lanes, line.radius_of_curvature, line.line_base_pos)
        save("lanes", image, lanes)
    except TypeError as e:
        print("Error:", e)

