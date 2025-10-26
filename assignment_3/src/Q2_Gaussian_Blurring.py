import skimage.io
from skimage.color import rgb2gray
import numpy as np
from matplotlib import pyplot as plt
import os

def create_gaussian_kernel(kernel_size, sigma):
    coords = np.arange(-(kernel_size//2), (kernel_size // 2) + 1)
    x, y = np.meshgrid(coords, coords, indexing='ij')

    kernel_gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    k = np.sum(kernel_gaussian)

    kernel_gaussian_normalized = kernel_gaussian / k

    return kernel_gaussian_normalized

def pad_kernel_for_dft(kernel, image_shape):
    """Pads a kernel for frequency domain filtering."""
    kernel_h, kernel_w = kernel.shape
    image_h, image_w = image_shape

    kernel_padded = np.zeros(image_shape)

    # 2. Find top-left coordinates to paste kernel for spatial centering
    paste_y = (image_h - kernel_h) // 2
    paste_x = (image_w - kernel_w) // 2

    # 3. Paste the kernel into the center
    kernel_padded[paste_y : paste_y + kernel_h, 
                  paste_x : paste_x + kernel_w] = kernel
    
    # 4. Shift quadrants to move kernel's center to (0,0)
    kernel_for_dft = np.fft.ifftshift(kernel_padded)
    return kernel_for_dft

def part_a(img_gray, output_dir_name):
    print("Running Part (a)...")
    img_shape = img_gray.shape

    # PART A
    sigma = 2.5
    kernel_size = 13
    kernel_g = create_gaussian_kernel(kernel_size, sigma)
    kernel_padded = pad_kernel_for_dft(kernel_g, img_shape)
    fft_img = np.fft.fft2(img_gray)
    fft_kernel = np.fft.fft2(kernel_padded)
    fft_blurred = fft_img * fft_kernel
    blur_img = np.fft.ifft2(fft_blurred)
    blur_img_real  = np.real(blur_img)


    plt.figure(figsize=(12,8))

    plt.subplot(121)
    plt.title("Original Image")
    plt.imshow(img_gray, cmap='gray')

    plt.subplot(122)
    plt.title("Image Blurred in Frequency Domain")
    plt.imshow(blur_img_real, cmap='gray')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir_name}/a_blurring_result.png")
    plt.show()

    print("Part (a) complete.")

    return kernel_g, kernel_padded, blur_img_real

def part_b(kernel_spatial_13, kernel_spatial_1036, output_dir_name):
    print("Running Part (b)...")

    epsilon = 1e-3
    # 13x13 DFT
    kernel_magnitude_13 = np.abs(np.fft.fftshift(np.fft.fft2(kernel_spatial_13)))
    log_13 = np.log( 1+ kernel_magnitude_13)

    # 13x13 DFT Inversed
    kernel_magnitude_13_inv = 1 / (kernel_magnitude_13 + epsilon)
    log_13_inv = np.log( 1+ kernel_magnitude_13_inv)
    
    # 1036x1036 DFT
    kernel_magnitude_1036 = np.abs(np.fft.fftshift(np.fft.fft2(kernel_spatial_1036)))
    log_1036 = np.log( 1+ kernel_magnitude_1036)

    # 1036x1036 DFT Inversed
    kernel_magnitude_1036_inv = 1 / (kernel_magnitude_1036 + epsilon)
    log_1036_inv = np.log( 1+ kernel_magnitude_1036_inv)

    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.title("i. 13x13 DFT Magnitude (Centered)")
    plt.imshow(log_13, cmap='gray', interpolation='nearest')
    plt.colorbar()

    plt.subplot(222)
    plt.title("ii. 13x13 Inverse Magnitude (Centered)")
    plt.imshow(log_13_inv, cmap='gray', interpolation='nearest')
    plt.colorbar()

    plt.subplot(223)
    plt.title("iii. 1036x1036 DFT Magnitude (Centered)")
    plt.imshow(log_1036, cmap='gray')
    plt.colorbar()

    plt.subplot(224)
    plt.title("iv. 1036x1036 Inverse Magnitude (Centered)")
    plt.imshow(log_1036_inv, cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f"{output_dir_name}/b_kernel_spectrums.png")
    plt.show()

    print("Part (b) complete.")

    return kernel_magnitude_1036, kernel_magnitude_1036_inv

def part_c(kernel_mag, output_dir_name):
    print("Running Part (c)...")

    M, N = kernel_mag.shape

    u = np.arange(M)
    v = np.arange(N)
    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
    U = u_grid - (M - 1) / 2
    V = v_grid - (N - 1) / 2
    sum_U_sq_V_sq = U**2 + V**2

    k_values = np.linspace(1e-6, 1e-3, 100) 
    min_error = np.inf
    best_k = 0

    for k in k_values:
        H_cont = np.exp(-k * sum_U_sq_V_sq)

        error = np.sum((H_cont - kernel_mag) ** 2)

        if error < min_error:
            min_error = error
            best_k = k
            
    print(f"Optimized k (k_opt) = {best_k}")

    H_cont_opt = np.exp(-best_k * sum_U_sq_V_sq)
    
    epsilon = 1e-3
    H_cont_opt_inv = 1 / (H_cont_opt + epsilon)

    plt.figure(figsize=(12, 8))

    plt.subplot(121)
    plt.title(f"i. Magnitude Spectrum of the Gaussian Fit |H_cont| (k_opt={best_k:.2e})")
    plt.imshow(np.log(1 + H_cont_opt), cmap='gray') 

    plt.subplot(122)
    plt.title("ii. Inverse 1 / (|H_cont| + eps)")
    plt.imshow(np.log(1 + H_cont_opt_inv), cmap='gray')

    plt.tight_layout()
    plt.savefig(f"{output_dir_name}/c_gaussian_fit_results.png")
    plt.show()

    print("Part (c) complete.")
    
    return H_cont_opt_inv

def part_d(original_image, blurred_image, H_direct_inv, H_fit_inv, output_dir_name):
    
    print("Running Part (d)...")

    fft_blurred = np.fft.fft2(blurred_image)
    fft_blurred_centered = np.fft.fftshift(fft_blurred)

    # Restore using the direct kernel DFT inverse
    fft_restored_direct = fft_blurred_centered * H_direct_inv
    restored_direct_ifft = np.fft.ifft2(np.fft.ifftshift(fft_restored_direct))
    restored_direct = np.real(restored_direct_ifft)

    # Restore using the Gaussian Fit inverse
    fft_restored_fit = fft_blurred_centered * H_fit_inv
    restored_fit_ifft = np.fft.ifft2(np.fft.ifftshift(fft_restored_fit))
    restored_fit = np.real(restored_fit_ifft)

    plt.figure(figsize=(15, 6))
    plt.subplot(131)
    plt.title("i. Original Image")
    plt.imshow(original_image, cmap='gray')

    plt.subplot(132)
    plt.title("ii. Restored (Direct Kernel Inverse)")
    plt.imshow(restored_direct, cmap='gray')

    plt.subplot(133)
    plt.title("iii. Restored (Gaussian Fit Inverse)")
    plt.imshow(restored_fit, cmap='gray')

    plt.tight_layout()
    plt.savefig(f"{output_dir_name}/d_restoration_results.png")
    plt.show()

    mse_direct = np.mean((original_image - restored_direct) ** 2)
    mse_fit = np.mean((original_image - restored_fit) ** 2)

    print("\nMSE Results")
    print(f"MSE (Direct Kernel Inverse): {mse_direct}")
    print(f"MSE (Gaussian Fit Inverse):  {mse_fit}")
    
    if mse_fit < mse_direct:
        print("The Gaussian Fit method gives better restoration.")
    else:
        print("The Direct Kernel Inverse method gives better restoration.")
        
    print("Part (d) complete.")

    return
def main():
    input_path = '../images/'
    output_dir_name = '../output/Q2/'
    os.makedirs(output_dir_name, exist_ok=True)

    img_rgb = skimage.io.imread(f"{input_path}buildings.jpg")
    img_gray = rgb2gray(img_rgb)
    
    # Part A:
    kernel_spatial_13, kernel_spatial_1036, blurred_img_real = part_a(img_gray, output_dir_name)

    # Part B:
    kernel_mag, kernel_mag_inv = part_b(kernel_spatial_13, kernel_spatial_1036, output_dir_name)

    # Part C:
    H_cont_inv = part_c(kernel_mag, output_dir_name)

    # Part D:
    part_d(img_gray, blurred_img_real, kernel_mag_inv, H_cont_inv, output_dir_name) 

    return

if __name__ == '__main__':
    main()