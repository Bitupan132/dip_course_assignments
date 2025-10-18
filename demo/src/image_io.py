import skimage.io
import skimage.color
import numpy as np
from matplotlib import pyplot as plt

def image_io():
    image_path = '../data/FRIENDS.jpg'
    image = skimage.io.imread(image_path)   #numpy array
    print(image.shape)
    print(np.min(image), np.max(image))

    # convert to [0,1]
    # image_zero_one = image.astype('double') / 255    #better to work in this range while processing
    # Or simply:
    image_zero_one = skimage.img_as_float(image)
    print(np.min(image_zero_one), np.max(image_zero_one))


    # dark image
    image_dark = image_zero_one * 0.5
    print(np.min(image_dark), np.max(image_dark))

    #display images in a single plot
    plt.subplot(131)
    plt.imshow(image)
    plt.subplot(132)
    plt.imshow(image_zero_one)
    plt.subplot(133)
    plt.imshow(image_dark)
    plt.show()

    # Save dark image to file
    output_path = '../data/FRIENDS_dark.jpg'
    image_dark = (image_dark * 255).astype('uint8')     # Convert to uint8 before saving to avoid warning.
    skimage.io.imsave(output_path, image_dark)

    return
def rgb_to_gray():
    image_path = '../data/FRIENDS.jpg'
    rgb_image = skimage.io.imread(image_path)

    gray_image = skimage.color.rgb2gray(rgb_image)
    print(np.max(rgb_image))

    plt.subplot(121)
    plt.imshow(rgb_image)
    plt.subplot(122)
    plt.imshow(gray_image, cmap='gray') 
    # cmap = 'gray when displaying a grayscale image, 
    # where each pixel is represented by a single intensity value (e.g., 0 for black, 255 for white), 
    # using cmap='gray' will render the image in its intended grayscale appearance. 
    # Without this parameter, Matplotlib might use a default colormap like 'viridis', 
    # resulting in a colorized representation of the grayscale data.
    plt.show()

    output_path = '../data/FRIENDS_gray.jpg'
    skimage.io.imsave(output_path, rgb_image)

    return

def main():
    # image_io()
    rgb_to_gray()
    return

if __name__ == '__main__':
    main()