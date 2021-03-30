# -*- coding: utf-8 -*-
"""
Created on 2021.3.24

@author: liu

from https://scikit-image.org/docs/dev/auto_examples/edges/plot_active_contours.html
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
from skimage.io import imread, imsave

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """
    def _store(x):
        lst.append(np.copy(x))

    return _store

if __name__ == '__main__':
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()

    img_color = imread('data/card.png')

    image = (imread('data/card.png', 1)*255)
    image = img_as_float(image)
    gimage = inverse_gaussian_gradient(image)

    # Initial level set
    init_ls = np.zeros(image.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1

    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_geodesic_active_contour(gimage, 200, init_ls,
                                               smoothing=1, balloon=-1.2,
                                               threshold=0.05,
                                               iter_callback=callback)
    ax[0].imshow(img_color)
    ax[0].set_title("original", fontsize=12)
    ax[1].imshow(img_color)

    ax[2].imshow(img_color, cmap="gray")
    ax[2].contour(ls, [0.5], colors='r')
    ax[2].set_title("result", fontsize=12)

    ax[3].imshow(img_color)
    ax[3].imshow(ls, cmap="gray")

    ax[3].set_axis_off()
    contour = ax[3].contour(evolution[0], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 0")
    contour = ax[3].contour(evolution[100], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 100")
    contour = ax[3].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 200")
    ax[3].legend(loc="upper right")
    ax[3].set_title('process', fontsize=12)

    for i in range(4): ax[i].set_axis_off()

    fig.tight_layout()
    plt.show()

    gif_list = []

    #save gif
    for i in range(11):
        plt.clf()
        plt.imshow(img_color)
        contour = plt.contour(evolution[20*i], [0.5], colors='y')
        plt.axis('off')
        plt.title("Iteration %d"%(20*i))
        plt.savefig('temp.png')
        gif_list.append(imread('temp.png'))

    img, *imgs = [Image.fromarray(gif_list[i]) for i in range(len(gif_list))]
    img.save(fp='snake.gif', format='GIF', append_images=imgs,
             save_all=True, duration=1000, loop=0)
