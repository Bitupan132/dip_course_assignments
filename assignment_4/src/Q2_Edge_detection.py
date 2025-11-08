import os
import skimage.io
from skimage.color import rgb2gray
import numpy as np
import cv2
from matplotlib import pyplot as plt

def edge_detection_sobel(image):

    M_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    M_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float64)

    I_x = cv2.filter2D(src= image, ddepth= cv2.CV_64F, kernel= M_x, borderType=cv2.BORDER_CONSTANT)
    I_y = cv2.filter2D(src= image, ddepth= cv2.CV_64F, kernel= M_y, borderType=cv2.BORDER_CONSTANT)

    gradient_mag = np.sqrt(I_x ** 2 + I_y ** 2)

    return gradient_mag

def edge_map_sobel(gradient_img, threshold):
    return ((gradient_img > threshold) * 255).astype(np.uint8)

def edge_map_LoG(log_image):
    log_center = log_image[:-1, :-1] # all pixels except the last row and last col
    log_right  = log_image[:-1, 1:]  # all pixels except last row and first col
    log_down   = log_image[1:, :-1]  # all pixels except first row and last col

    check_horizontal = log_center * log_right < 0
    check_vertical   = (log_center * log_down < 0)
    edge_map_bool = check_horizontal | check_vertical

    edge_map = np.zeros_like(log_image, dtype=np.uint8)
    edge_map[:-1, :-1][edge_map_bool] = 255

    return edge_map
    

def laplacian_of_gaussian_kernel(kernel_size, sigma):
    coords = np.arange(-(kernel_size//2), (kernel_size // 2) + 1)
    x, y = np.meshgrid(coords, coords, indexing='ij')

    kernel_LoG = ((x**2 + y**2 - 2 * sigma**2) / (sigma**4)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    return kernel_LoG
def gaussian_kernel(kernel_size, sigma):
    coords = np.arange(-(kernel_size//2), (kernel_size // 2) + 1)
    x, y = np.meshgrid(coords, coords, indexing='ij')

    kernel_gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    return kernel_gaussian / np.sum(kernel_gaussian)

def plot_results_2a(images, title, output_dir):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Part 2a: Clean Image Edge Detection - {title.capitalize()}', fontsize=18)
    
    titles = ['Clean Original', 'Clean - Sobel (2a)', 'Clean - LoG (2a)']
    keys = ['clean', 'clean_sobel', 'clean_log']
    
    for ax, subplot_title, key in zip(axes, titles, keys):
        if key in images:
            ax.imshow(images[key], cmap='gray', vmin=0, vmax=255)
            ax.set_title(subplot_title, fontsize=14)
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{output_dir}a/{title}_results_2a.png')
    plt.close(fig)

def plot_results_2b(images, title, output_dir):

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Part 2b: Noisy/Smoothed Edge Detection - {title.capitalize()}', fontsize=18)
    
    titles = [
        'Noisy', 'Noisy + Smoothed (2b)',
        'Noisy + Smoothed - Sobel (2b)', 'Noisy + Smoothed - LoG (2b)'
    ]
    keys = [
        'noisy', 'noisy_smoothed',
        'noisy_smoothed_sobel', 'noisy_smoothed_log'
    ]
    
    for ax, subplot_title, key in zip(axes.flat, titles, keys):
        if key in images:
            ax.imshow(images[key], cmap='gray', vmin=0, vmax=255)
            ax.set_title(subplot_title, fontsize=14)
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{output_dir}b/{title}_results_2b.png')
    plt.close(fig)

def plot_comparison_for_2c(images, title, output_dir):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Edge Detection Comparison: {title.capitalize()}', fontsize=20)

    titles = [
        'Clean Original', 'Clean - Sobel (2a)', 'Clean - LoG (2a)',
        'Noisy', 'Noisy - Sobel', 'Noisy - LoG',
        'Noisy + Smoothed (2b)', 'Noisy + Smoothed - Sobel (2b)', 'Noisy + Smoothed - LoG (2b)'
    ]

    keys = [
        'clean', 'clean_sobel', 'clean_log',
        'noisy', 'noisy_sobel', 'noisy_log',
        'noisy_smoothed', 'noisy_smoothed_sobel', 'noisy_smoothed_log'
    ]

    for ax, subplot_title, key in zip(axes.flat, titles, keys):
        if key in images:
            ax.imshow(images[key], cmap='gray', vmin=0, vmax=255)
            ax.set_title(subplot_title)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    plt.savefig(f'{output_dir}c/{title}_comparison_plot.png')
    plt.close(fig)

def main():
    input_dir = '../data/images/'
    output_dir = '../data/output/Q2/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}a/', exist_ok=True)
    os.makedirs(f'{output_dir}b/', exist_ok=True)
    os.makedirs(f'{output_dir}c/', exist_ok=True)

    image_files = {
        'checkerboard': 'Checkerboard.png',
        'coins': 'Coins.png',
        'flowers': 'flowers.png',
        'mainbuilding': 'MainBuilding.png'
    }
    images_dict = {}

    for name, filename in image_files.items():
        img = skimage.io.imread(f'{input_dir}{filename}')
        images_dict[name] = img

    # Part 2a LoG parameters
    log_params_a = {
        'checkerboard': (13, 7),
        'coins': (7, 0.5),
        'flowers': (7, 1),
        'mainbuilding': (11, 0.5)
    }
    # Part 2b LoG parameters
    log_params_b = {
        'checkerboard': (13, 0.5),
        'coins': (11, 2),
        'flowers': (15, 2),
        'mainbuilding': (13, 2)
    }

    gauss_k_b = gaussian_kernel(7, 3)
    noise_mean = 0
    noise_sigma = 15
    sobel_threshold_a = 120
    sobel_threshold_b = 120

    print("Running 2a, 2b, and 2c for each image...")
    for name, img in images_dict.items():
        print(f"Processing all tasks for: {name}")
        
        plot_images = {}
        plot_images['clean'] = img
        
        # PART A:

        # 2a(i): Sobel on Clean
        sobel_clean = edge_map_sobel(edge_detection_sobel(img), sobel_threshold_a)
        plot_images['clean_sobel'] = sobel_clean
        skimage.io.imsave(f'{output_dir}a/{name}_sobel_edge.png', sobel_clean)
        
        # 2a(ii): LoG on Clean
        k_a, s_a = log_params_a[name]
        log_k_a = laplacian_of_gaussian_kernel(k_a, s_a)
        log_img_a = cv2.filter2D(src=img, ddepth=cv2.CV_64F, kernel=log_k_a, borderType=cv2.BORDER_REPLICATE)
        log_clean = edge_map_LoG(log_img_a)
        plot_images['clean_log'] = log_clean
        skimage.io.imsave(f'{output_dir}a/{name}_log_edge.png', log_clean)

        plot_results_2a(plot_images, name, output_dir)
        
        #PART B:

        # Add Gaussian Noise
        noise = np.random.normal(noise_mean, noise_sigma, img.shape)
        noisy_image = np.clip(img.astype(np.float64) + noise, 0, 255)
        plot_images['noisy'] = noisy_image.astype(np.uint8)
        skimage.io.imsave(f'{output_dir}b/{name}_noisy.png', plot_images['noisy'])

        # Apply Gaussian Smoothing
        smoothed_image = cv2.filter2D(src=noisy_image, ddepth=cv2.CV_64F, kernel=gauss_k_b, borderType=cv2.BORDER_CONSTANT)
        plot_images['noisy_smoothed'] = smoothed_image.astype(np.uint8)
        skimage.io.imsave(f'{output_dir}b/{name}_noisy_smoothed.png', plot_images['noisy_smoothed'])

        # Edge detection on smoothed noisy image
        # Sobel
        sobel_noisy_smoothed = edge_map_sobel(edge_detection_sobel(smoothed_image), sobel_threshold_b)
        plot_images['noisy_smoothed_sobel'] = sobel_noisy_smoothed
        skimage.io.imsave(f'{output_dir}b/{name}_noisy_smoothed_sobel_edge.png', sobel_noisy_smoothed)
        
        # LoG
        k_b, s_b = log_params_b[name]
        log_k_b = laplacian_of_gaussian_kernel(k_b, s_b)
        log_img_b = cv2.filter2D(src=smoothed_image, ddepth=cv2.CV_64F, kernel=log_k_b, borderType=cv2.BORDER_REPLICATE)
        log_noisy_smoothed = edge_map_LoG(log_img_b)
        plot_images['noisy_smoothed_log'] = log_noisy_smoothed
        skimage.io.imsave(f'{output_dir}b/{name}_noisy_smoothed_log_edge.png', log_noisy_smoothed)
        
        plot_results_2b(plot_images, name, output_dir)

        # PART C:

        # Sobel on Noisy (unsmoothed)
        sobel_noisy = edge_map_sobel(edge_detection_sobel(noisy_image), sobel_threshold_b)
        plot_images['noisy_sobel'] = sobel_noisy
        
        # LoG on Noisy (unsmoothed) 
        log_img_noisy = cv2.filter2D(src=noisy_image, ddepth=cv2.CV_64F, kernel=log_k_a, borderType=cv2.BORDER_REPLICATE)
        log_noisy = edge_map_LoG(log_img_noisy)
        plot_images['noisy_log'] = log_noisy
        
        plot_comparison_for_2c(plot_images, name, output_dir)
    return

if __name__ == '__main__':
    main() 