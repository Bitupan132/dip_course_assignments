import skimage.io
from skimage.color import rgb2gray
import numpy as np
from matplotlib import pyplot as plt
import os
from helper_functions import get_blur_image

def sharpen(image:np.ndarray, p = 0):
    if len(image.shape) == 3:
        image  = rgb2gray(image)*255

    if p<0 or p>1:
        print("Error. 'p' should be between 0 and 1")
        return image.astype(np.uint8)

    if p == 0:
        return image.astype(np.uint8)
    
    factor = p*10
    kernel_size = 5

    image = image.astype(np.float32)

    blur_image = get_blur_image(image=image, kernel_size=kernel_size)
    blur_image = blur_image.astype(np.float32)

    mask = image - blur_image

    sharp_image = image + (factor * mask)

    final_image = np.clip(sharp_image, 0, 255).astype(np.uint8)

    return final_image

def main():
    input_path = '../images/'
    output_dir_name = '../output/Q3/'
    os.makedirs(output_dir_name, exist_ok=True)

    p_values = [0, 0.1, 0.2, 0.5, 0.75, 1]

    study_image = skimage.io.imread(f'{input_path}study.png')
    study_sharpened_images = {}
    for p in p_values:
        study_sharpened_images[f"sharp_image_{'_'.join(str(p).split('.'))}"] = sharpen(image=study_image, p=p)

    plt.figure(figsize=(10,6))
    i =1
    for p in p_values:
        key = f"sharp_image_{'_'.join(str(p).split('.'))}"
        sharp_image = study_sharpened_images[key]
        plt.subplot(2,3, i)
        plt.title(f'sharp image (p = {p})')
        plt.imshow(sharp_image, cmap='gray')
        i = i+1

    plt.tight_layout()
    plt.savefig(f"{output_dir_name}/sharpened_study.png")
    plt.show()

    input_images = ['fox.png', 'owl.png']
    for image_name in input_images:
        input_file = f'{input_path}{image_name}'
        image = skimage.io.imread(input_file)
        for p in p_values:
            output_file = f"{output_dir_name}sharp_{image_name.split('.')[0]}_{'_'.join(str(p).split('.'))}.png"
            sharp_image = sharpen(image=image, p=p)
            skimage.io.imsave(output_file, sharp_image)

    return

if __name__ == '__main__':
    main()