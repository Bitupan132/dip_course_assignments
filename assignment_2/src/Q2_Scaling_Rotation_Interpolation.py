import skimage.io
from skimage.color import rgb2gray
import numpy as np
from matplotlib import pyplot as plt
import os
from helper_functions import pad_to_square, pad_for_rotation

def bilinear_interpolation(image, a, b):
    m = int(np.floor(a))
    n = int(np.floor(b))

    dx = a - m
    dy = b - n

    # Get the four neighbors 
    # since using padded image add 1 to the locations
    # I00 = image[m+1, n+1]
    # I01 = image[m+1, n+2]
    # I10 = image[m+2, n+1]
    # I11 = image[m+2, n+2]
    I00 = image[m, n]
    I01 = image[m, n+1]
    I10 = image[m+1, n]
    I11 = image[m+1, n+1]  

    return (
                (1 - dx) * (1 - dy) * I00 +
                dx * (1 - dy) * I10 +
                (1 - dx) * dy * I01 +
                dx * dy * I11
            )

def upsample_image(image, scale=2):
    padded_image = np.pad(image, pad_width=1, mode='edge')

    height_in, width_in = image.shape
    height_out, width_out = height_in * scale, width_in * scale

    upsampled_image = np.zeros((height_out, width_out), dtype=image.dtype)
    
    for x in range(height_out):
        for y in range(width_out):
            a = x / scale
            b = y / scale
            upsampled_image[x, y] = bilinear_interpolation(padded_image, a, b)
    return upsampled_image

def rotate_image(image, angle = 45, scale = 1):
    padded_image = pad_for_rotation(image=image, angle=45, scale = scale)

    # height, width = image.shape
    height, width = padded_image.shape

    # rotated_image = np.zeros_like(image)
    rotated_image = np.zeros_like(padded_image)

    angle_rad = np.deg2rad(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    c_x, c_y = (height) / 2, (width) / 2
    for i in range(height):
        for j in range(width):
            # Shift to center
            i_c = i - c_x
            j_c = j - c_y

            # CCW
            a_c = i_c * cos_theta + j_c * sin_theta
            b_c = -i_c * sin_theta + j_c * cos_theta
            
            a = a_c + c_x
            b = b_c + c_y

            # if (a, b) is inside the input image
            if 0 <= a < height - 1 and 0 <= b < width - 1:
                rotated_image[i, j] = bilinear_interpolation(padded_image, a, b)

            # If mapped coordinate is outside, set to 0 (black)
            else:
                rotated_image[i, j] = 0
    return rotated_image


def main():
    input_path = '../images/flowers.png'
    image = skimage.io.imread(input_path)
    output_dir_name = '../output/Q2'
    os.makedirs(output_dir_name, exist_ok=True)
    
    rotated_image = rotate_image(image=image, angle=45)
    upsampled_rotateed_image = upsample_image(image=rotated_image, scale=2)
    # print(upsampled_rotateed_image.shape)

    skimage.io.imsave(f'{output_dir_name}/upsampled_rotateed_image.png', upsampled_rotateed_image)
    skimage.io.imsave(f'{output_dir_name}/rotated_image.png', rotated_image)

    upsampled_image = upsample_image(image=image, scale=2)
    rotated_upsampled_image = rotate_image(image=upsampled_image, angle=45, scale=2)
    # print(rotated_upsampled_image.shape)

    skimage.io.imsave(f'{output_dir_name}/upsampled_image.png', upsampled_image)
    skimage.io.imsave(f'{output_dir_name}/rotated_upsampled_image.png', rotated_upsampled_image)

    diff1 = rotated_upsampled_image.astype(np.float32) - upsampled_rotateed_image.astype(np.float32)
    print("Before clipping the pixels intensities of diff:")
    print(f"Min:{np.min(diff1)}, Max: {np.max(diff1)}")
    diff = np.clip(diff1, 0, 255).astype(np.uint8)
    print("After clipping the pixels intensities of diff:")
    print(f"Min:{np.min(diff)}, Max: {np.max(diff)}")
    skimage.io.imsave(f'{output_dir_name}/diff.png', diff)


    plt.figure(figsize=(10,6))
    plt.subplot(221)
    plt.title('Original')
    plt.imshow(image, cmap='gray')
    plt.subplot(222)
    plt.title('rotated_upsampled_image')
    plt.imshow(rotated_upsampled_image, cmap='gray')
    plt.subplot(223)
    plt.title('upsampled_rotateed_image')
    plt.imshow(upsampled_rotateed_image, cmap='gray')
    plt.subplot(224)
    plt.title('diff')
    plt.imshow(diff, vmin=0, vmax=255,cmap='gray')
    plt.tight_layout()
    plt.show()

    return

if __name__ == '__main__':
    main() 