import skimage.io
from skimage.color import rgb2gray
import cv2
import numpy as np
from matplotlib import pyplot as plt
from helper_functions import get_otsu_threshold, get_binarized_image, get_histogram
import os

def process_image(image:np.ndarray, kernel_size, optimal_variances):

    blur_kernel = np.ones((kernel_size,kernel_size), dtype=np.float32) / (kernel_size)**2

    blur_image = cv2.filter2D(src=rgb2gray(image), ddepth= -1, kernel=blur_kernel, borderType=cv2.BORDER_CONSTANT)
    blur_image = (blur_image * 255).astype('uint8')

    histogram_blur_image =  get_histogram(blur_image)

    otsu_threshold, withing_class_var = get_otsu_threshold(blur_image)
    binarized_image = get_binarized_image(image=blur_image, threshold=otsu_threshold)

    optimal_variances[kernel_size] = withing_class_var

    print()
    print(f"For Kernel Size: {kernel_size}")
    print(f"Threshold at which Within-Class Variance is Optimal: {otsu_threshold}")
    print(f"Optimal within-class variance is: {withing_class_var}")

    return blur_image, histogram_blur_image, binarized_image, optimal_variances


def main():
    input_path = '../images/moon_noisy.png'
    image = skimage.io.imread(input_path)
    output_dir_name = '../output/Q1'
    os.makedirs(output_dir_name, exist_ok=True)

    kernel_sizes = [5, 29, 129]
    optimal_variances = {}

    plt.figure(figsize=(15,10))
    i=1

    for kernel_size in kernel_sizes:
        blur_image, histogram_blur_image, binarized_image, optimal_variances = process_image(
            image=image, 
            kernel_size=kernel_size, 
            optimal_variances=optimal_variances)
        
        plt.subplot(3,3,i)
        plt.title(f'Blur Image ({kernel_size} x {kernel_size})')
        plt.imshow(blur_image, cmap='gray')
        plt.subplot(3,3,i+1)
        plt.title(f'Binarized Blur Image ({kernel_size} x {kernel_size})')
        plt.imshow(binarized_image, cmap='gray')
        plt.subplot(3,3,i+2)
        plt.title(f'Histogram of Blur Image ({kernel_size} x {kernel_size})')
        plt.plot(histogram_blur_image)
        i = i+3

    plt.tight_layout()
    plt.savefig(f"{output_dir_name}/compared_output.png")
    plt.show()

    optimal_kernel_size = min(optimal_variances, key=optimal_variances.get)
    print()
    print(f"Optimal filter size that minimizes the within-class variance is: {optimal_kernel_size}")
    return

if __name__ == '__main__':
    main() 