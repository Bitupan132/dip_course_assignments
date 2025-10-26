import numpy as np
from matplotlib import pyplot as plt
import os

def compute_2d_dft(img):
    fft_img = np.fft.fft2(img)
    fft_centered = np.fft.fftshift(fft_img)
    return fft_centered

def a_generate_image_using_sinusoids():
    M = 256

    m = np.arange(M)
    n = np.arange(M)
    # 2D grid of coordinates
    m_grid, n_grid = np.meshgrid(m, n, indexing='ij')
    # The m_grid holds the row coordinate for each pixel:    | 0 | 0 | 0 | |---|---|---| | 1 | 1 | 1 | | 2 | 2 | 2 |
    # The n_grid holds the column coordinate for each pixel: | 0 | 1 | 2 | |---|---|---| | 0 | 1 | 2 | | 0 | 1 | 2 |

    # x1 = np.ndarray(shape=(M,M), dtype=np.float64)
    x1 = np.sin(2 * np.pi * 12 * m_grid / M)
    x2 = np.sin(2 * np.pi * 8 * n_grid / M)
    x3 = np.sin(2 * np.pi * (6 * m_grid + 10 * n_grid) / M)

    x = (x1 + x2 + x3) / 3

    fft_x = compute_2d_dft(x)

    n = 5
    log_magnitude_x_5 = np.log(1 + np.abs(fft_x)**(1/n))
    n = 30
    log_magnitude_x_30 = np.log(1 + np.abs(fft_x)**(1/n))

    output_dir_name = '../output/Q1/'
    os.makedirs(output_dir_name, exist_ok=True)

    plt.figure(figsize=(12,8))

    plt.subplot(321)
    plt.title("x1")
    plt.imshow(x1, cmap='gray')

    plt.subplot(322)
    plt.title("x2")
    plt.imshow(x2, cmap='gray')

    plt.subplot(323)
    plt.title("x3")
    plt.imshow(x3, cmap='gray')

    plt.subplot(324)
    plt.title("x")
    plt.imshow(x, cmap='gray')

    plt.subplot(325)
    plt.title("magnitude of DFT image of x (n = 5)")
    plt.imshow(log_magnitude_x_5, cmap='gray')

    plt.subplot(326)
    plt.title("magnitude of DFT image of x (n = 30)")
    plt.imshow(log_magnitude_x_30, cmap='gray')

    plt.tight_layout()
    plt.savefig(f"{output_dir_name}/results_combined.png")
    plt.show()

    return x, fft_x, log_magnitude_x_30

def b_create_directional_filter(angles_array, theta_min, theta_max):

    filter_h = (angles_array >= theta_min) & (angles_array <= theta_max)

    return filter_h.astype(int)

def b_create_angle_array(M = 256):
    u = np.arange(M)
    v = np.arange(M)

    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')

    u_centered = u_grid - M/2
    v_centered = v_grid - M/2

    # Calculate angle for every pixel
    angles_rad = np.arctan2(v_centered, u_centered)
    angles_deg = np.degrees(angles_rad)
    # angles_deg: 256 x 256, each pixel contains its angle from the center.
    return angles_deg

def b_apply_directional_filter(fft_img, filter_h):
    # 1. apply filter to fft_img (centered) - fft_filtered - freq spectrum
    fft_img_filtered = fft_img * filter_h

    # 2. inverse dft
    fft_img_filtered_inverse = np.fft.ifftshift(fft_img_filtered)
    fft_img_filtered_inverse = np.fft.ifft2(fft_img_filtered_inverse)
    fft_filtered_inverse_real = np.real(fft_img_filtered_inverse)

    return fft_img_filtered, fft_filtered_inverse_real

def b_plots_for_filter(x, log_magnitude_x, filter_mask, fft_filtered_img, reconstructed_image, filter_name):
    output_dir_name = '../output/Q1/'
    os.makedirs(output_dir_name, exist_ok=True)

    plt.figure(figsize=(12,8))

    plt.subplot(231)
    plt.title("i. Original Image")
    plt.imshow(x, cmap='gray')

    plt.subplot(232)
    plt.title("ii. Original Spectrum")
    plt.imshow(log_magnitude_x, cmap='gray')

    plt.subplot(233)
    plt.title(f"iii. Filter: {filter_name}")
    plt.imshow(filter_mask, cmap='gray')

    plt.subplot(234)
    plt.title("iv. Filtered Spectrum")
    plt.imshow(np.log(1 + np.abs(fft_filtered_img)**(1/30)), cmap='gray')

    plt.subplot(235)
    plt.title("v. Reconstructed Image")
    plt.imshow(reconstructed_image, cmap='gray')

    plt.tight_layout()
    plt.savefig(f"{output_dir_name}/plots_for_filter_{filter_name}.png")
    plt.show()


    return

def main():

    print("A. Creating images from sinusoids...")
    x, fft_x, log_magnitude_x = a_generate_image_using_sinusoids()
    print("Finished creating images from sinusoids...")

    angles_array = b_create_angle_array()

    H1 = b_create_directional_filter(angles_array, -20, 20)
    H2 = b_create_directional_filter(angles_array, 70, 110)
    H3 = b_create_directional_filter(angles_array, 25, 65)
    H4 = np.maximum(np.maximum(H1, H2), H3)

    filters_to_run = [(H1, "H1"),(H2, "H2"),(H3, "H3"),(H4, "H4")]
    mses = []

    print("B. Starting directional filtering...")

    for filter_mask, filter_name in filters_to_run:
        print(f"Processing filter: {filter_name}")
    
        # 1. Apply the filter
        fft_filtered_spectrum, reconstructed_image = b_apply_directional_filter(fft_x, filter_mask)
        
        # 2. Plot the results
        b_plots_for_filter(x, log_magnitude_x, filter_mask, fft_filtered_spectrum, reconstructed_image, filter_name)

        mse = np.mean((x - reconstructed_image)**2)
        mses.append(("filter_name", mse))

    print("Finished processing all filters.")

    print("C. Calculating MSEs:")
    for item in mses:
        print(f"MSE between original (x) and {item[0]}_reconstructed: {item[1]}")

    return

if __name__ == '__main__':
    main()