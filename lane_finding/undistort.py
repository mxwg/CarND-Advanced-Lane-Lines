import cv2
import numpy as np

mtx = None
dist = None

def _load_calibration():
    """Load the camera calibration file (assumed to be in the directory above this package."""
    global mtx, dist
    if mtx is None or dist is None:
        import pickle
        import os
        cwd = os.path.dirname(os.path.abspath(__file__))
        calibration_file = os.path.join(cwd, '../camera_calibration.p')
        with open(calibration_file, 'rb') as f:
            (mtx, dist) = pickle.load(f)
        print("Imported camera calibration.")


def undistort(image):
    """Undistorts an image using a pre-computed camera calibration."""
    assert image.shape[2] == 3, "only three-channel images are supported!"
    if mtx is None or dist is None:
        _load_calibration()
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    return undistorted

def warp(image, src, dst, inverse=False):
    """Warps an image given two sets of points."""
    # calculate the perspective Transform
    M = cv2.getPerspectiveTransform(src, dst)
    if inverse:
        M = cv2.getPerspectiveTransform(dst, src)
    image_size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    return warped

def warp_to_lane(image, inverse=False):
    """Warps the lane region of an image to a straight region."""
    assert image.shape[0] == 720 and image.shape[1] == 1280
    top = 450
    lane     = np.float32([[205, image.shape[0]], [1110, image.shape[0]], [692-6, top], [586+10, top]])
    straight = np.float32([[300, image.shape[0]], [900, image.shape[0]], [900, 0], [300, 0]])
    return warp(image, lane, straight, inverse)

