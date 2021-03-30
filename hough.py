# -*- coding: utf-8 -*-
"""
Created on 2021.3.24

@author: liu

reference https://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from analysis import analysis

def get_hough(image, num=20):
    edges = canny(image, sigma=2, low_threshold=5, high_threshold=50)

    # Detect two radii
    hough_radii = np.arange(20, 100, 1)
    print(hough_radii)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=num)

    return edges, zip(cy, cx, radii)

if __name__ == '__main__':
    # Load picture and detect edges
    image = (imread('data/cell.png', 1)*255).astype('uint8')

    gray_edges, gray_result = get_hough(image)

    fig, axes = plt.subplots(2, 2)
    ax = axes.flatten()

    _, _, _, labeled1 = analysis(image)
    labeled1[labeled1>0] = 255
    label_edges, label_result = get_hough(labeled1)

    ax[0].imshow(image, cmap='gray')
    for center_y, center_x, radius in gray_result:
        circle = plt.Circle((center_x, center_y), radius, color='blue', fill=False)
        ax[0].add_patch(circle)
    ax[1].imshow(gray_edges)
    ax[2].imshow(labeled1, cmap='gray')
    for center_y, center_x, radius in label_result:
        circle = plt.Circle((center_x, center_y), radius, color='blue', fill=False)
        ax[2].add_patch(circle)
    ax[3].imshow(label_edges)

    ax[0].set_title('gray input', fontsize=12) 
    ax[1].set_title('gray sobel', fontsize=12) 
    ax[2].set_title('msk input', fontsize=12) 
    ax[3].set_title('msk sobel', fontsize=12) 
    for i in range(4): ax[i].set_axis_off()
    fig.tight_layout()
    plt.show()

