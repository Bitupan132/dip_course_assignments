import os
import skimage.io
from skimage.color import rgb2gray
import numpy as np
import cv2
from matplotlib import pyplot as plt

def a_image_downsample(img, factor):
    if len(img.shape) == 3:
        img = rgb2gray(img)

    output_img = img[::factor, ::factor ]
    return output_img

def b_create_gaussian_kernel(kernel_size, sigma):
    coords = np.arange(-(kernel_size//2), (kernel_size // 2) + 1)
    x, y = np.meshgrid(coords, coords, indexing='ij')

    kernel_gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    k = np.sum(kernel_gaussian)

    kernel_gaussian_normalized = kernel_gaussian / k

    return kernel_gaussian_normalized
def plot_downsampled_images(downsampling_factors, output_imgs_raw, output_imgs_preprocessed, output_imgs_sk_resize, output_dir):
    num_factors = len(downsampling_factors)
    num_methods = 3 
    
    fig, axes = plt.subplots(nrows=num_factors, ncols=num_methods, figsize=(15, 5 * num_factors))
    
    # Set the titles for the columns (only need to do this for the first row)
    axes[0, 0].set_title('Raw Downsampling (Part a)', fontsize=14)
    axes[0, 1].set_title('Gaussian Blur + Downsample (Part b)', fontsize=14)
    axes[0, 2].set_title('skimage.resize (Library)', fontsize=14)

    for i, factor in enumerate(downsampling_factors):
        
        img_raw = output_imgs_raw[i]
        img_pre = output_imgs_preprocessed[i]
        img_lib = output_imgs_sk_resize[i]

        axes[i, 0].imshow(img_raw, cmap='gray', vmin=0, vmax=255)
        axes[i, 0].set_ylabel(f'Factor = {factor}', fontsize=14)
        axes[i, 0].axis('off') 
        

        axes[i, 1].imshow(img_pre, cmap='gray', vmin=0, vmax=255)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(img_lib, cmap='gray', vmin=0, vmax=255)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show() # Display the plot

    # You can also save this figure for your report
    fig.savefig(f'{output_dir}/comparison_plot.png')
def main():

    input_file_path = '../data/images/city.png'
    output_dir = '../data/output/Q1'
    os.makedirs(output_dir, exist_ok=True)

    input_img = skimage.io.imread(input_file_path)

    # Part A:
    downsampling_factors = [2,4,5]
    output_imgs_raw = []
    for factor in downsampling_factors:
        output_img_raw = a_image_downsample(input_img, factor)
        output_imgs_raw.append(output_img_raw)
        skimage.io.imsave(f'{output_dir}/raw_downsample_factor_{factor}.png', output_img_raw)

    # Part B:
    # kernel_gaussian = b_create_gaussian_kernel(5, 2)
    # gaussian_low_pass_img = cv2.filter2D(src=input_img, ddepth=-1, kernel=kernel_gaussian, borderType=cv2.BORDER_CONSTANT)
    gaussian_low_pass_img = cv2.GaussianBlur(input_img, (5,5), 2)
    # print(np.min(gaussian_low_pass_img), np.max(gaussian_low_pass_img))
    output_imgs_preprocessed = []
    for factor in downsampling_factors:
        output_img_preprocessed = a_image_downsample(gaussian_low_pass_img, factor)
        output_imgs_preprocessed.append(output_img_preprocessed)
        skimage.io.imsave(f'{output_dir}/preprocessed_downsample_factor_{factor}.png', output_img_preprocessed)
    
    # output_imgs_cv_resize = []
    # for factor in downsampling_factors:
    #     output_img_cv_resize = cv2.resize(input_img, dsize=None, fx=1/factor, fy=1/factor, interpolation=cv2.INTER_AREA)
    #     output_imgs_cv_resize.append(output_img_cv_resize)
    #     skimage.io.imsave(f'{output_dir}/cv_resize_downsample_factor_{factor}.png', output_img_cv_resize)
    

    output_imgs_sk_resize = []
    for factor in downsampling_factors:
        output_h, output_w = input_img.shape[0]//factor, input_img.shape[1]//factor
        output_img_sk_resize = (skimage.transform.resize(input_img, output_shape=(output_h, output_w), anti_aliasing=True) * 255).astype(np.uint8)
        output_imgs_sk_resize.append(output_img_sk_resize)
        skimage.io.imsave(f'{output_dir}/sk_resize_downsample_factor_{factor}.png', output_img_sk_resize)

    plot_downsampled_images(downsampling_factors, output_imgs_raw, output_imgs_preprocessed, output_imgs_sk_resize, output_dir)

    # PART C:
    factor = 5
    output_h, output_w = input_img.shape[0]//factor, input_img.shape[1]//factor
    target_img = skimage.transform.resize(input_img, output_shape=(output_h, output_w), anti_aliasing=True)
    
    window_sizes = [3, 5, 7, 9, 11, 13, 15, 17]
    sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    min_error = np.inf
    opt_window = 0
    opt_sigma = 0

    for window_size in window_sizes:
        for sigma in sigmas:
            low_pass_img = cv2.GaussianBlur(input_img, (window_size,window_size), sigma)
            output_img = a_image_downsample(low_pass_img, factor) / 255
            mse = np.mean((output_img - target_img) ** 2)
            if mse < min_error:
                min_error = mse
                opt_window  = window_size
                opt_sigma = sigma

    print("The optimal window size and sigma value that minimize the MSE:")
    print(f"Window size: {opt_window}, Sigma: {opt_sigma}")

    return

if __name__ == '__main__':
    main() 