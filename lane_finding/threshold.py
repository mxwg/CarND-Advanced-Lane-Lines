import cv2
import numpy as np

def sobelx(img, t_min, t_max):
    """Apply the sobel operator in the x direction and return a binary image."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= t_min) & (scaled_sobel <= t_max)] = 255
    return binary

def threshold(img, t_min, t_max):
    """Return the pixels of the binary image between min and max."""
    binary = np.zeros_like(img)
    binary[(img >= t_min) & (img <= t_max)] = 255
    return binary

def get_channel_in_color_space(img, space, channel):
    """Convert the given image to the color space specified and return one channel."""
    img_conv = cv2.cvtColor(img, getattr(cv2, "COLOR_RGB2" + space.upper()))
    img_chan = img_conv[:,:,space.find(channel)]
    normalized = img_chan #np.uint8(255*img_chan/np.max(img_chan)) # (necessary for jupyter)
    return normalized

def combine_sparse(images, limit=15*1e6):
    """Combines the images in the list if the are sparse enough."""
    combined = np.zeros_like(images[0])
    for img in images:
        if np.sum(img) < limit:
            combined += img
    return combined

def threshold_basic(img):
    """Threshold a grayscale image and return the binary image."""
    assert len(img.shape) == 3, "only color images are supported"

    # create a binary image using the sobel operator
    sobel = sobelx(img, 20, 200)

    # create binary images of different tresholded color channels
    hsv_v = threshold(get_channel_in_color_space(img, 'hsv', 'v'), 210, 255)
    hls_s = threshold(get_channel_in_color_space(img, 'hls', 's'), 110, 255)
    hls_l = threshold(get_channel_in_color_space(img, 'hls', 'l'), 140, 250)

    # combine the thresholded images
    color = combine_sparse([hsv_v, hls_s, hls_l])

    # combine the thresholded and sobel images
    combined_binary = combine_sparse([color, sobel], limit=20*1e6)

    assert np.sum(combined_binary) > 0
    return combined_binary

