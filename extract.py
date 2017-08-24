import os
import glob
import matplotlib.image as mplimg
import numpy as np
import cv2
import shutil
from lane_finding.undistort import undistort, warp_to_lane
from lane_finding.threshold import threshold_basic
from lane_finding.fit_lines import fit_lanes, plot_windows, plot_lanes, augment_image_with_lane
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
output_dir = "extracted_images"

print("Cleaning up output directory {}...".format(output_dir))
try:
    shutil.rmtree(output_dir)
except FileNotFoundError:
    pass
os.mkdir(output_dir)
counter = 0

def process_image(img):
    global counter
    mplimg.imsave(os.path.join(output_dir, "output_{:05d}.jpg".format(counter)), img)
    counter += 1
    return img

filename = "project_video_augmented.mp4"
#filename = "project_video.mp4"


clip1 = VideoFileClip(filename)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color

white_clip.write_videofile(os.path.join(output_dir, "deleteme.mp4"), audio=False)

os.remove(os.path.join(output_dir, "deleteme.mp4"))
print("Done.")
