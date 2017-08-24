import cv2
import numpy as np

def sob(img, t_min, t_max):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= t_min) & (scaled_sobel <= t_max)] = 255
    return binary

def thresh(img, t_min, t_max):
    binary = np.zeros_like(img)
    binary[(img >= t_min) & (img <= t_max)] = 255
    return binary

def conv(img, space, channel):
    #print(np.max(img))
    img_conv = cv2.cvtColor(img, getattr(cv2, "COLOR_RGB2" + space.upper()))
    img_chan = img_conv[:,:,space.find(channel)]
    normalized = img_chan#np.uint8(255*img_chan/np.max(img_chan))
    #normalized = np.uint8(255*img_chan/np.max(img_chan))
    return normalized

def comb(images, limit=15*1e6):
    combined = np.zeros_like(images[0])
    for img in images:
        if np.sum(img) < limit:
            combined += img
    return combined

def threshold_basic(img):
    """Threshold a grayscale image and return the binary image."""
    assert len(img.shape) == 3, "only color images are supported"

    sobel = sob(img, 20, 200)
    #sobel = sob(img, 20, 100)

    one = thresh(conv(img, 'hsv', 'v'), 210, 255)
    two = thresh(conv(img, 'hls', 's'), 110, 255)
    three = thresh(conv(img, 'hls', 'l'), 140, 250)
    color = comb([one, two, three])

    combined_binary = comb([color, sobel], limit=20*1e6)
    if np.sum(combined_binary) == 0:
        combined_binary = sobel
    assert np.sum(combined_binary) > 0
    return combined_binary

