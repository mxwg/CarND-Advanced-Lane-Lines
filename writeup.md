# **Advanced Lane Finding Project**

[//]: # (Image References)
[pattern]: ./images/pattern_distorted.png "Distorted pattern"
[pattern_undistorted]: ./images/pattern_undistorted.png "Undistorted pattern"
[original]: ./images/0_original_output_00016.jpg
[undist]: ./images/1_undist_output_00016.jpg
[warped]: ./images/2_warped_output_00027.jpg
[saturation]: ./images/saturation_bad.png
[combined]: ./images/3_binary_warped_output_00027.jpg
[sliding_window]: ./images/4_curves_output_00017.jpg
[tracking]: ./images/4_curves_output_00016.jpg
[augmented]: ./images/5_lanes_output_00027.jpg

The goal of this project is to find lanes in a video stream and annotate it with the detected lanes, 
the curvature of the lanes and the distance of the vehicle to the center of the lane.

## Camera Calibration
The camera calibration was done using the Jupyter notebook `camera-calibration.ipynb`.

All calibration images were loaded and converted to grayscale.
Then the pattern points were detected for 17 of the 20 images using the OpenCV function `findChessboardCorners`.
The camera matrix and distortion coefficients were computed using `calibrateCamera` and saved to disk using `pickle`.

An example of a calibration image and its undistorted version can be seen below.
![Distorted pattern][pattern]
![Undistorted pattern][pattern_undistorted]


## Lane Detection Pipeline

### Undistortion
The first step of the pipeline is to indistort the images entering the pipeline using the previously computed camera matrix and distortion coefficients (in `lane_finding/undistort.py`, line 20).
![Original image][original]
![Undistorted image][undist]


### Bird's Eye View
The undistorted image is next warped to a bird's eye view of the lanes using the function `warp_to_lane` (in `lane_finding/undistort.py`, line 38).
Internally, the OpenCV functions `getPerspectiveTransform` and `warpPerspective` are used for this.
The result of this transformation can be seen below.
[Bird's Eye View of the lanes][warped]


### Thresholded Binary Images
The next step of the pipeline is to create binary images that contain as much information about the lanes themselves as possible but no unnecessary clutter.

This step is implemented in `lane_finding/threshold.py`.

The function `threshold_basic` (line 35) first applies the Sobel operator in x direction to a grayscale version of the image.
Then three color thresholded images are created.
One uses the V channel of the HSV color space, the other two the S and L channels of the HLS color space.
The image below shows a scene where the usually good HSV/V channel does not pick up the left lane very well.
The HLS/S and HLS/L both pick up the lane, but the HLS/S channel also contains a lot of spurious pixels, which happens when there are large patches of differently colored lane surface.

This problem is handled in the combined image by combining only binary thresholded images that don't contain these large spurious regions (`combine_sparse`, line 48).
![Combination of different color thresholded images][saturation]

The last step is to combine the Sobel image with the binary thresholded image.
The result can be seen below.
![The final thresholded image][combined]


### Lane pixel fitting
The next step of the pipeline is fitting a second order polynomial to the pixels identified as belonging to a lane.
This is done using functions in the file `lane_finding/fit_lines.py`.

There are two approaches to fitting the polynomial, `fit_lanes` uses the sliding window approach demonstrated in the lecture, while `track_lanes` uses a previously fit polynomial to constrain the search space.

Visualizations of the result of both functions can be seen below.

This first image shows the sliding window approach (line 29).
First, a histogram of the pixels in the lower half of the binary image is created.
From the histogram, the peaks corresponding to the left and right lanes are extracted and used as a starting point for the sliding windows.
If a window contains enough pixels, the average of these pixels define the center of the next window above.
In this way, the windows slide from the bottom to the top of the image, following the pixels making up a lane.
All pixels identified as belonging to a lane in this way are collected and a polynomial is fit to both the left and right lanes.

This is a visualization of the sliding window detector.
The windows are shown in green, the left lane in red and the right lane in blue.
The fit polynomials are shown in yellow.

Additionally, the visualization shows red, green and yellow circles near the bottom.
The red circles indicate the interception of the yellow polynomial with the bottom of the image, the green circle is the ideal center position of the lane, the yellow circle is the actual center of the lane.
The green lanes are tracked averages, as explained below.
![Sliding window approach][sliding_window]

The second visualization shows lane tracking (`track_lanes`, line 90).
There are no windows, as the search space is constrained by the previous polynomial (shown in transparent green).
The other elements of the visualization are as in the sliding window example.
![Tracking approach][tracking]


### Curvature and Position of the Vehicle
The curvature is calculated after either the sliding window or the tracking approach have fitted polynomials.
The function used is `calculate_curvature_in_meters` in `lane_finding/fit_lines.py`, line 23.
This is based on the approach shown in the lecture.

The offset of the vehicle is calculated in `Line._compute_offset` (line 77 in `lane_finding/line.py`).

Both values are calculated in meters and written to the augmented image (see below).


### Augmented Lane Image
The final step of the pipeline is using the identified lanes to overlay the original undistorted image with the lanes.

An example of this augmented image is shown below.
![The final augmented image][augmented]


## Video Pipeline
For annotating a video, the pipeline saves its results in the `Line` class (`lane_finding/line.py`, line 8).

This class keeps a running average of the last 5 fitted polynomials (line 15).
With each new fit, there is a check (line 58) to see whether the new polynomial results in similar intersection coordinates at the bottom of the image.
If the difference is small enough (line 66), the new measurement is added to the running list (line 87).

The class provides an average (line 47) of the last polynomials, curvatures and distances that is used in the annotation of the images.


## Discussion
problems issues
A lot of time for this project was spent on evaluating different parameters for thresholding.
The current combination handles many cases but with further time could certainly be improved.

The tracking of the lane polynomials is very basic at the moment.
Some easy improvements would be to use an exponential moving average, which would give greater weight to more recent detections.
Another could be to use e.g. a Kalman filter for estimating the polynomials.

The pipeline will fail as soon as the images differ too much from the ones that were used to manually tune the thresholding parameters.
Having annotated ground truth images would make it possible to automate the discovery of these parameters, e.g. with some machine learning.


