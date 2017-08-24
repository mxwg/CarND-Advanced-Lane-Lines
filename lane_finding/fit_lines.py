import cv2
import numpy as np
from lane_finding.undistort import warp_to_lane

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
margin = 100
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def curve_m(x,y):
    y_eval = 720
    fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad

def fit_lanes(binary_warped):
    nwindows = 9
    minpix = 50
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    window_height = np.int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    windows = []
    left_lane_inds = []
    right_lane_inds = []
    leftx_current = leftx_base
    rightx_current = rightx_base
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        windows.append((win_xleft_low,win_y_low))
        windows.append((win_xleft_high,win_y_high))
        windows.append((win_xright_low,win_y_low))
        windows.append((win_xright_high,win_y_high))
        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    curve_left = curve_m(leftx, lefty)
    curve_right = curve_m(rightx, righty)

    lx = left_fit[2]
    rx = right_fit[2]
    c = lx + (rx-lx)/2
    cm = (640-c) * xm_per_pix
    #print("neft {} opt {} center {} right {} -> {}".format(lx,\
            #640, c, rx, cm))

    return left_fit, right_fit, left_lane_inds, right_lane_inds, curve_left,\
            curve_right, windows

def track_lanes(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    curve_left = curve_m(leftx, lefty)
    curve_right = curve_m(rightx, righty)

    lx = left_fit[2]
    rx = right_fit[2]
    c = lx + (rx-lx)/2
    cm = (640-c) * xm_per_pix
    #print("teft {} opt {} center {} right {} -> {}".format(lx,\
            #640, c, rx, cm))

    return left_fit, right_fit, left_lane_inds, right_lane_inds, \
            curve_left, curve_right, []

def plot_windows(out_img, w):
    for i in range(0, len(w), 4):
        cv2.rectangle(out_img, w[i+0], w[i+1], GREEN, 2)
        cv2.rectangle(out_img, w[i+2], w[i+3], GREEN, 2)
    return out_img


def plot_lanes(binary_warped, left_fit, right_fit, left_lane_inds,\
               right_lane_inds, tracked=False):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if tracked:
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        window_img = np.zeros_like(out_img)
        cv2.fillPoly(window_img, np.int_([left_line_pts]), GREEN)
        cv2.fillPoly(window_img, np.int_([right_line_pts]), GREEN)
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # color left and right lanes
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = RED
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = BLUE

    # plot lane centers
    points_left = np.int32(np.stack([left_fitx, ploty], axis=1))
    points_right = np.int32(np.stack([right_fitx, ploty], axis=1))
    cv2.polylines(out_img, [points_left], False, YELLOW, thickness=2)
    cv2.polylines(out_img, [points_right], False, YELLOW, thickness=2)

    # draw offset
    lx = left_fit[0]*720**2 + left_fit[1]*720 + left_fit[2]
    rx = right_fit[0]*720**2 + right_fit[1]*720 + right_fit[2]
    c = int(lx + (rx-lx)/2)
    cv2.circle(out_img, (int(lx), 720-2*5), 5, RED, 5)
    cv2.circle(out_img, (int(rx), 720-2*5), 5, RED, 5)
    cv2.circle(out_img, (640, 720-2*5), 5, GREEN, 5)
    cv2.circle(out_img, (c, 720-2*5), 5, YELLOW, 5)

    return out_img

def plot_lanes_only(out_img, binary_warped, left_fit, right_fit):
    #nonzero = binary_warped.nonzero()
    #nonzeroy = np.array(nonzero[0])
    #nonzerox = np.array(nonzero[1])
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # plot lane centers
    points_left = np.int32(np.stack([left_fitx, ploty], axis=1))
    points_right = np.int32(np.stack([right_fitx, ploty], axis=1))
    cv2.polylines(out_img, [points_left], False, GREEN, thickness=2)
    cv2.polylines(out_img, [points_right], False, GREEN, thickness=2)

    return out_img

def get_radii_m(binary_warped, left_lane_inds, right_lane_inds, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    lefty = nonzeroy[left_lane_inds]
    righty = nonzeroy[right_lane_inds]
    leftx = nonzerox[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    y_eval = 720
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    print(left_fit_cr)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad)

def write_text(img, radius, dist):
    txt = "Radius of Curvature: {:5.0f} m, Vehicle is {:.2f} m {} of the center.".format(
    radius, abs(dist), "left" if dist < 0.0 else "right")
    cv2.putText(img, txt, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    return img

def info(img, txt):
    cv2.putText(img, txt, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    return img

def augment_image_with_lane(image, left_fit, right_fit):
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    color_warp = np.zeros_like(image).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    #newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    newwarp = warp_to_lane(color_warp, inverse=True)
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result
