import skimage.io
import numpy as np
import sys
from matplotlib import pyplot as plt
from collections import Counter
from Q2_Otsus_Binarization import get_binarized_image, get_within_class_variance
import os

def get_otsu_binarization_threshold(image: np.ndarray) -> int:
    min_within_class_var = sys.float_info.max
    min_threshold = 0

    for threshold in range(256):
        within_class_var_at_t = get_within_class_variance(image = image, threshold=threshold)
        if within_class_var_at_t < min_within_class_var:
            min_within_class_var = within_class_var_at_t
            min_threshold = threshold
    return min_threshold

def compute_block_positions(image_shape, block_size, overlap):
    step = block_size - overlap
    positions = []
    total_rows = image_shape[0]
    total_cols = image_shape[1]

    # collect positions that contains the NxN blocks
    # there will be some pixels on the right side and on the bottom that are left out
    for row in range(0, total_rows - block_size + 1, step):
        for col in range(0, total_cols - block_size + 1, step):
            positions.append((row, col))

    # Edge Cases:
    # A. collect positions that contains the bottom side's left out pixels' blocks. 
    # for 512x512 image and 250x250 block size these will be: (262,0),(262,200)
    for col in range(0, total_cols - block_size + 1, step):
        positions.append((total_rows - block_size, col))

    # B. collect positions that contains the right side's left out pixels' blocks. 
    # for 512x512 image and 250x250 block size these will be: (0,262),(200,262)
    for row in range(0, total_rows - block_size + 1, step):
        positions.append((row, total_cols - block_size))
    
    # C. Still the bottom right block, (262,262), is left out:
    positions.append((total_rows - block_size, total_cols - block_size))

    # Bacause of the edge case handling logic there might be duplicates.
    positions = list({p for p in positions})
    positions.sort(key=lambda position: (position[0],position[1]))

    return positions

# Step 2: Accumulate binarized block votes for each pixel
def process_blocks(image, block_size, overlap):
    positions = compute_block_positions(image.shape, block_size, overlap)
    # print(positions)

    # To collect votes, we'll use a list of lists vote_list
    # vote_list is 512 rows, 512 cols
    # each element holds a list.
    # each item in that list is either 0 or 255.
    # e.g if vote_list[0][200] = [0, 255, 0]: it implies that the pixel (0,200) is 0 in one binarized block, 255 in another, 0 in another.
    vote_lists = [[[] for _ in range(image.shape[1])] for _ in range(image.shape[0])]

    for (row, col) in positions:
        image_block = image[row:row+block_size, col:col+block_size]
        # threshold = skimage.filters.threshold_otsu(image_block)
        threshold = get_otsu_binarization_threshold(image_block)
        binarized_image_block = get_binarized_image(image_block, threshold)
        # print(np.max(binarized_image_block))

        for i in range(block_size):
            for j in range(block_size):
                vote_lists[row + i][col + j].append(binarized_image_block[i, j])

    # Now, resolve each pixel by majority vote: 0 or 1 in case of tie?: handle according to the original image?
    output_image = np.zeros(image.shape, dtype=np.uint8)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if vote_lists[row][col]:
                most_common = Counter(vote_lists[row][col]).most_common()
                # if no overlap or not a tie
                if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                    output_image[row, col] = most_common[0][0]
                # if tie: assign 1.
                else:
                    # print(f"Tie. votes: {votes}, most common: {most_common}. Actual: {image[row][col]}")
                    output_image[row, col] = 1
            else:
                output_image[row, col] = image[row, col]

    return output_image


def adaptive_binarization(image, block_size, overlap, output_path):
    result = process_blocks(image, block_size, overlap)
    result = (result * 255).astype('uint8')
    skimage.io.imsave(output_path, result) 
    return result

def main():
    image_path = '../data/sudoku.png'
    output_dir_name = '../output/Q3'
    os.makedirs(output_dir_name, exist_ok=True)

    image = skimage.io.imread(image_path)
    block_sizes = [512, 50, 25, 10, 5]

    plt.figure(figsize=(10,6))
    plt.subplot(2,3,1)
    plt.title('Original Image')    
    plt.imshow(image, cmap='gray')
    i = 2
    for block_size in block_sizes:
        print(f"--- Running for block size {block_size}x{block_size}")
        overlap = int(0.2 * block_size)
        output_image_path = f'{output_dir_name}/sudoku_adaptive_binarized_{block_size}.png'
        final_image = adaptive_binarization(image, block_size, overlap, output_image_path)
        plt.subplot(2,3,i)
        if block_size == 512:
            plt.title(f'Global Binarized Image')   
        else:
            plt.title(f'{block_size}x{block_size} Adaptive Binarized Image')    
        plt.imshow(final_image, cmap='gray')   
        i+=1
        print(f"\n--- {block_size}x{block_size} Complete ---\n")
    plt.savefig(f'{output_dir_name}/all_compared.png')
    plt.show()

    return

if __name__ == '__main__':
    main()

