import numpy as np
from skimage import io
import os
from matplotlib import pyplot as plt
from Q3_Adaptive_Binarization import get_otsu_binarization_threshold
from Q2_Otsus_Binarization import get_binarized_image

def dsu_find(parent, i):
    if parent[i] == i:
        return i
    parent[i] = dsu_find(parent, parent[i])
    return parent[i]

def dsu_union(parent, i, j):
    root_i = dsu_find(parent, i)
    root_j = dsu_find(parent, j)
    if root_i != root_j:
        if root_i < root_j:
            parent[root_j] = root_i
        else:
            parent[root_i] = root_j


def binarize_image(image: np.ndarray):
    threshold = get_otsu_binarization_threshold(image=image)
    binarized_image = np.ones_like(image)
    binarized_image = image < threshold
    # binarized_image = get_binarized_image(image=image, threshold=threshold)
    # binarized_image = 1 - binarized_image
    print(f"  Threshold: {threshold:.3f}, Foreground pixels: {binarized_image.sum()}")
    plt.figure()
    plt.imshow(binarized_image, cmap='gray')
    plt.show()
    return binarized_image

# Replacing skimage.color.gray2rgb(img_as_ubyte(gray_image))
def convert_gray_to_rgb(gray_image):
    # Convert to uint8
    gray_image = (gray_image * 255).astype(np.uint8)
    # convert to rgb
    rgb_image = np.stack([gray_image, gray_image, gray_image], axis=2)
    return rgb_image

def get_connected_components(binarized_image):

    padded_image = np.pad(binarized_image, pad_width=1, mode='constant',constant_values=False)
    labels = np.zeros_like(padded_image, dtype=np.int32)
    current_label = 1
    dsu_parent = np.arange(padded_image.shape[0] * padded_image.shape[1] // 2)

    for row in range(1, padded_image.shape[0] - 1):
        for col in range(1, padded_image.shape[1] - 1):

            # if it is a foreground pixel: 1
            if padded_image[row, col]:

                # get the 4 previous neighnours
                neighbors = [
                    labels[row - 1, col - 1],  # Top-left
                    labels[row - 1, col],      # Top
                    labels[row - 1, col + 1],  # Top-right
                    labels[row, col - 1]       # Left
                ]

                # Neighbors that are foreground pixels
                foreground_neighbors = [n for n in neighbors if n > 0]

                # New Region
                if not foreground_neighbors:
                    labels[row, col] = current_label
                    current_label += 1
                # Connected region
                else:
                    min_label = min(foreground_neighbors)
                    labels[row, col] = min_label
                    for label in foreground_neighbors:
                        if label != min_label:
                            dsu_union(dsu_parent, min_label, label)

    # Remove padding
    labels = labels[1:-1, 1:-1]

    # Resolve equivalences
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    # print(f"unique labels: {unique_labels}")

    root_labels = np.array([dsu_find(dsu_parent, l) for l in unique_labels])
    # print(f"root labels: {root_labels}")

    relabel_map = np.zeros(labels.max() + 1, dtype=labels.dtype)
    relabel_map[unique_labels] = root_labels
    # print(f"relabel_map : {relabel_map}")

    final_labels = relabel_map[labels]
    # print(f"final labels : {final_labels}")

    return final_labels

def find_largest_character_component(labels):
    component_labels, counts = np.unique(labels[labels > 0], return_counts=True)
    largest_component_label = component_labels[np.argmax(counts)]

    print(f"  Total components: {len(component_labels)}")
    print(f"  Largest component: {largest_component_label} ({np.max(counts)} pixels)")

    return largest_component_label

def create_highlighted_image(gray_image, labels, largest_component_label):
    mask = (labels == largest_component_label)
    highlighted_image = convert_gray_to_rgb(gray_image) 
    # Highlight the largest component in red
    highlighted_image[mask] = [255, 0, 0]

    return highlighted_image

def get_largest_character(gray_image: np.ndarray):
    # Normalize to [0, 1]
    # gray_image = gray_image / 255
    # Binarize the Image
    binarized_image = binarize_image(gray_image)

    labels = get_connected_components(binarized_image=binarized_image)
    largest_component_label = find_largest_character_component(labels=labels)
    highlighted_image = create_highlighted_image(gray_image/255,labels, largest_component_label)

    return highlighted_image

def main():
    image_path = '../data/quote.png'
    output_dir = '../output/Q4'

    image = io.imread(image_path)
    highlighted_image = get_largest_character(image)

    os.makedirs(output_dir, exist_ok=True)
    io.imsave(f'{output_dir}/quote_highlighted.png',highlighted_image)

    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.subplot(122)
    plt.title("Highlighted Largest Connected Component")
    plt.imshow(highlighted_image)
    plt.show()

if __name__ == '__main__':
    main()