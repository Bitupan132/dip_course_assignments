import skimage.io
import numpy as np
from Q1_Histogram_Computation import get_histogram
import sys
from matplotlib import pyplot as plt
import os

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

def get_between_class_variance(image: np.ndarray, threshold: int) -> float:

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
    
    # overall_class_mean  = (prob_class_0 * mean_class_0) + (prob_class_1 * mean_class_1)

    # between_class_variance = (prob_class_0 * ((mean_class_0 - overall_class_mean) ** 2)) + (prob_class_1 * ((mean_class_1 - overall_class_mean) ** 2))
    # return between_class_variance
    return (prob_class_0 * prob_class_1) * ((mean_class_0  - mean_class_1) ** 2)

def get_binarized_image(image: np.ndarray, threshold: int) -> np.ndarray:
    binarized_image = np.zeros_like(image)

    # binarized_image[image <= threshold] = 0
    # binarized_image[image > threshold] = 1
    binarized_image = image > threshold

    return binarized_image

def q2a_otsus_binarization_within_class_variance(output_directory):
    image_path = '../data/coins.png'
    im = skimage.io.imread(image_path)

    min_within_class_var = sys.float_info.max
    min_threshold = 0

    for threshold in range(256):
        within_class_var_at_t = get_within_class_variance(image = im, threshold=threshold)
        # print(f"Within-Class Variance at Threshold {threshold}: {within_class_var_at_t}")
        if within_class_var_at_t < min_within_class_var:
            min_within_class_var = within_class_var_at_t
            min_threshold = threshold
    print()
    print("Solution to Question 2A:")
    print(f"Threshold at which Within-Class Variance is Minimum: {min_threshold}")
    print(f"Minimum within-class variance is: {min_within_class_var}")

    #Binarize
    bin_im = get_binarized_image(im, min_threshold)
    bin_im = (bin_im * 255).astype('uint8')
    output_path = f'{output_directory}/coins_q2a_binarized.png'
    skimage.io.imsave(output_path, bin_im)
    return bin_im
    
def q2b_otsus_binarization_between_class_variance(output_directory):
    image_path = '../data/coins.png'
    im = skimage.io.imread(image_path)
    im = im.astype(np.int16)        # to prevent overflow while doing offset. otherwise 255+20 will be 19

    modified_image = im + 20
    modified_image = np.clip(modified_image, 0, 255).astype(np.uint8)

    # min_within_class_var = sys.float_info.max
    # min_threshold = 0
    # for threshold in range(256):
    #     within_class_var_at_t = get_within_class_variance(image = modified_image, threshold=threshold)
    #     # print(f"Within-Class Variance at Threshold {threshold}: {within_class_var_at_t}")
    #     if within_class_var_at_t < min_within_class_var:
    #         min_within_class_var = within_class_var_at_t
    #         min_threshold = threshold
    # print(f"Threshold at which Within-Class Variance is Minimum: {min_threshold}")
    # print(f"Minimum within-class variance is: {min_within_class_var}")
    # O/P:
    # Threshold at which Within-Class Variance is Minimum: 145
    # Minimum within-class variance is: 255.93054363922533

    max_between_class_var = sys.float_info.min
    max_threshold = 0
    for threshold in range(256):
        between_class_var_at_t = get_between_class_variance(image = modified_image, threshold=threshold)
        # print(f"Between-Class Variance at Threshold {threshold}: {between_class_var_at_t}")
        if between_class_var_at_t > max_between_class_var:
            max_between_class_var = between_class_var_at_t
            max_threshold = threshold
    print()
    print("Solution to Question 2B:")
    print(f"Threshold at which Between-Class Variance is Maximum: {max_threshold}")
    print(f"Maximum between-class variance is: {max_between_class_var}")
    # Threshold at which Between-Class Variance is Maximum: 145
    # Maximum between-class variance is: 2852.9562808051214

    #Binarize
    bin_im = get_binarized_image(modified_image, max_threshold)
    bin_im = (bin_im * 255).astype('uint8')
    output_path = f'{output_directory}/coins_q2b_binarized.png'
    skimage.io.imsave(output_path, bin_im)

    return bin_im

def main():
    output_dir_name = '../output/Q2'
    os.makedirs(output_dir_name, exist_ok=True)

    binarized_image_within = q2a_otsus_binarization_within_class_variance(output_dir_name)
    binarized_image_between = q2b_otsus_binarization_between_class_variance(output_dir_name)

    plt.figure(figsize=(15,10))
    plt.subplot(1,3,1)
    plt.title('Original Image')    
    plt.imshow(skimage.io.imread('../data/coins.png'), cmap='gray')
    plt.subplot(1,3,2)
    plt.title('Binarized Image using Within Class Variance')    
    plt.imshow(binarized_image_within, cmap='gray')
    plt.subplot(1,3,3)
    plt.title('Binarized Image using Between Class Variance')    
    plt.imshow(binarized_image_between, cmap='gray')
    plt.savefig(f"{output_dir_name}/compared_output.png")
    plt.show()

    return 
if __name__ == '__main__':
    main()