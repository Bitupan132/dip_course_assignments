import numpy as np
import sys
from skimage.color import rgb2gray
import cv2

def get_histogram(image: np.ndarray) -> list:
    height, width = image.shape
    freq = [0] * 256

    for i in range(height):
        for j in range(width):
            intensity = image[i][j]
            freq[intensity] += 1

    return freq

def get_normalized_histogram(histogram: list) -> list:
    total_pixels = np.sum(histogram)
    n_hist = [histogram[i]/total_pixels for i in range(len(histogram))]
    return n_hist

def get_class_probabilities(normalized_histogram: list, threshold: int):
    cls_0 = np.sum(normalized_histogram[:threshold+1])
    cls_1 = np.sum(normalized_histogram[threshold+1:])
    return cls_0, cls_1

def get_class_means(normalized_histogram: list, threshold: int, prob_class_0: float, prob_class_1: float): 
    sum_0 = 0
    sum_1 = 0
    for k in range(0,threshold+1):
        sum_0 += k * normalized_histogram[k]
    for k in range(threshold+1, 256):
        sum_1 += k * normalized_histogram[k]

    return sum_0/prob_class_0 if prob_class_0>0 else 0, sum_1/prob_class_1 if prob_class_1>0 else 0

def get_class_variances(normalized_histogram: list, threshold: int, prob_class_0: float, prob_class_1: float, mean_class_0: float, mean_class_1: float):
    sum_0 = 0
    sum_1 = 0
    for k in range(0,threshold+1):
        sum_0 += ((k - mean_class_0) ** 2) * normalized_histogram[k]
    for k in range(threshold+1, 256):
        sum_1 += ((k - mean_class_1) ** 2) * normalized_histogram[k]

    return sum_0/prob_class_0 if prob_class_0>0 else 0, sum_1/prob_class_1 if prob_class_1>0 else 0

def get_within_class_variance(image: np.ndarray, threshold: int) -> float:

    hist = get_histogram(image=image)

    normalized_histogram = get_normalized_histogram(histogram=hist)

    prob_class_0, prob_class_1 = get_class_probabilities(
        normalized_histogram = normalized_histogram, 
        threshold = threshold
    )
    
    mean_class_0, mean_class_1 = get_class_means(
        normalized_histogram = normalized_histogram, 
        threshold = threshold, 
        prob_class_0 = prob_class_0, 
        prob_class_1 = prob_class_1
    )
    
    var_class_0, var_class_1 = get_class_variances(
        normalized_histogram = normalized_histogram, 
        threshold = threshold, 
        prob_class_0 = prob_class_0, 
        prob_class_1 = prob_class_1, 
        mean_class_0 = mean_class_0, 
        mean_class_1 = mean_class_1
    )

    return (prob_class_0 * var_class_0) + (prob_class_1 * var_class_1)

def get_binarized_image(image: np.ndarray, threshold: int) -> np.ndarray:
    """Returns Binarized image [0,1] in floating point"""
    binarized_image = np.zeros_like(image)
    binarized_image = image > threshold
    return binarized_image

def get_otsu_threshold(image: np.ndarray) -> int:
    min_within_class_var = sys.float_info.max
    min_threshold = 0

    for threshold in range(256):
        within_class_var_at_t = get_within_class_variance(image = image, threshold=threshold)
        if within_class_var_at_t < min_within_class_var:
            min_within_class_var = within_class_var_at_t
            min_threshold = threshold

    return min_threshold, min_within_class_var

def pad_to_square(image):
    height, width = image.shape

    square_length = int(np.ceil(np.sqrt(height**2 + width**2)))

    pad_height = square_length - height
    pad_width = square_length - width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded_image = np.pad(
        image,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',constant_values=0
    )
    return padded_image

def pad_for_rotation(image, angle, scale=1):
    angle_rad = np.deg2rad(angle)
    
    # Using original dimensions by dividing by scale before calculation
    height, width = image.shape
    original_height = height / scale
    original_width = width / scale

    cos_theta = np.abs(np.cos(angle_rad))
    sin_theta = np.abs(np.sin(angle_rad))
    
    # Calculating the required size based on the ORIGINAL dimensions
    new_width = int(np.ceil(original_width * cos_theta + original_height * sin_theta))
    new_height = int(np.ceil(original_width * sin_theta + original_height * cos_theta))

    # scaling up the calculated size
    final_width = new_width * scale
    final_height = new_height * scale

    square_length = max(final_width, final_height)

    # Calculating padding based on the CURRENT image size
    pad_height = square_length - height
    pad_width = square_length - width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded_image = np.pad(
        image,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant', constant_values=0
    )
    return padded_image

def get_blur_image(image:np.ndarray, kernel_size: int):
    blur_kernel = np.ones((kernel_size,kernel_size), dtype=np.float32) / (kernel_size)**2

    blur_image = cv2.filter2D(src=image, ddepth= -1, kernel=blur_kernel, borderType=cv2.BORDER_REPLICATE)
    # blur_image = (blur_image).astype('uint8')

    return blur_image