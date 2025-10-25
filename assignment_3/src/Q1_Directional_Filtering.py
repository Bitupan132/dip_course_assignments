import skimage.io
from skimage.color import rgb2gray
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

    plt.figure(figsize=(15,10))

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

    return 


def main():

    a_generate_image_using_sinusoids()

    return

if __name__ == '__main__':
    main()