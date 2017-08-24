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
from lane_finding.fit_lines import get_radii_m, get_offset, write_text, info

# Get images
input_folder = "test_images/track3"
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
from lane_finding.lane import Line
line = Line()
for image in image_names:
    print("working on image {}".format(image))
    try:
        img = mplimg.imread(image)
        img = img[:,:,:3]
        undist = undistort(img)
        #save("undist", image, undist)
        warped = warp_to_lane(undist)
        #save("a_warped", image, warped)
        gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
        #save("gray", image, gray, cmap='gray')
        #binary =  threshold_basic(undist)
        #save("binary", image, binary, cmap='gray')
        #binary_warped = warp_to_lane(binary)
        binary_warped = threshold_basic(warped)
        #save("binary_warped", image, binary_warped, cmap='gray')
        if not line.detected:
            print("Full detection...")
            left_fit, right_fit, left_lane_inds, right_lane_inds, cl, cr, windows = fit_lanes(binary_warped)
            print("fit", left_fit)
            line.update(left_fit, right_fit, cl, cr, force=True)
        else:
            left_fit, right_fit, left_lane_inds, right_lane_inds, cl, cr, windows = \
            track_lanes(binary_warped, line.best_fit_left, line.best_fit_right)
            line.update(left_fit, right_fit, cl, cr)

        curves = plot_lanes(binary_warped, left_fit, right_fit,\
                            left_lane_inds, right_lane_inds, line.detected)
        curves = plot_windows(curves, windows)
        curves = info(curves, "detected" if line.detected else "not detected")
        if line.best_fit_left is not None:
            curves = plot_lanes_only(curves, binary_warped,\
                                     line.best_fit_left, line.best_fit_right)
        save("curves", image, curves)
        #lanes = augment_image_with_lane(undist, left_fit, right_fit)
        lanes = augment_image_with_lane(undist, line.best_fit_left,\
                                        line.best_fit_right)
        #dist = get_offset_m(binary_warped, left_lane_inds, right_lane_inds)
        dist = get_offset(line.best_fit_left, line.best_fit_right)
        txt = "Radius: {:5.2f} m\nRadius: {:5.2f} m\nVehicle is {:.2f} m {} of the center.".format(
        cl, cr, abs(dist), "left" if dist < 0.0 else "right")
        print(txt)
        lanes = write_text(lanes, line.radius_of_curvature, dist)
        save("lanes", image, lanes)
    except TypeError as e:
        print("Error:", e)

