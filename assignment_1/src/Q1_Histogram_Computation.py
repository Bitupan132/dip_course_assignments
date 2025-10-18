import skimage.io
import numpy as np
from matplotlib import pyplot as plt
import os

def get_histogram(image: np.ndarray) -> list:
    # print(image.shape)
    # print(np.min(image), np.max(image))
    height, width = image.shape
    freq = [0] * 256

    for i in range(height):
        for j in range(width):
            intensity = image[i][j]
            freq[intensity] += 1

    return freq

def plot_hisogram(frequencies: list):
    os.makedirs('../output/Q1', exist_ok= True)
    plt.figure(figsize=(10,6))
    plt.plot(frequencies)
    plt.xlabel('Intensity Level')
    plt.ylabel('Frequency')
    plt.title('Frequency vs Intensity Level')
    plt.savefig('../output/Q1/histogram_coins.png')
    plt.show()
    return

def compute_avg(frequencies: list): # -> int:
    sum = 0
    total_pixels = np.sum(frequencies)
    for i in range(len(frequencies)):
       sum += i * frequencies[i]
    avg = sum / total_pixels
    return avg

def main():
    image_path = '../data/coins.png'
    im = skimage.io.imread(image_path)

    freq = get_histogram(image=im)
    # print(frequencies)
    plot_hisogram(frequencies=freq)
    avg_from_hist = compute_avg(frequencies=freq)
    avg_actual = np.mean(im)
    print(f"Average intensity of the image using histogram: {avg_from_hist}")
    print(f"Actual average intensity: {avg_actual}")
    print(f"Are the average intensity from histogram and actual intensity same: {avg_actual == avg_from_hist} ")
    return

if __name__ == '__main__':
    main()