# -*- coding: utf-8 -*-
"""
Created on 2021.3.24

@author: liu
"""
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import numpy as np
from scipy.ndimage import binary_fill_holes, label, generate_binary_structure, distance_transform_edt
from skimage.morphology import watershed
from skimage.measure import regionprops
from skimage.filters import  sobel

def analysis(img):
    img_binary=img.copy()
    # threshold
    img_binary = np.array([1]*200+[0]*55)[img_binary]
    # binary fill
    img_filled = binary_fill_holes(img_binary)

    #label 
    labeled = label(img_filled, generate_binary_structure(2, 1))[0].astype('uint8')
    ls= regionprops(labeled)
    lst = np.array([0]*(len(ls)+1))  

    #filter the small object               
    for i in ls:
        if i.area<1100: lst[i.label]=0
        else: lst[i.label] = i.label
    labeled1 = lst[labeled.astype(np.int32)]

    #relabel
    labeled1 = label(labeled1, generate_binary_structure(2, 1))[0].astype('uint8')

    return img_binary, img_filled, labeled, labeled1
if __name__ == '__main__':
    img = (imread('data/cell.png', 1)*255).astype('uint8')

    img_binary, img_filled, labeled, labeled1 = analysis(img)

    ls= regionprops(labeled1)
    lst = np.array([0]*(len(ls)+1)) 

    #print the idx and due attributes     
    for i in ls: print(i.label, i.area, i.local_centroid)

    fig, axes = plt.subplots(2, 3)
    ax = axes.flatten()

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('original', fontsize=12)
    ax[1].imshow(img_binary)
    ax[1].set_title('binary fill', fontsize=12)
    ax[2].imshow(img_filled)
    ax[2].set_title('labeled', fontsize=12)
    ax[3].imshow(labeled)
    ax[3].set_title('labeled', fontsize=12)
    ax[4].imshow(labeled1)
    ax[4].set_title('area filter', fontsize=12)
    ax[5].imshow(sobel(labeled1>0)*400+img)
    ax[5].set_title('sobel edge on image', fontsize=12)
    
    for i in range(6): ax[i].set_axis_off()

    fig.tight_layout()
    plt.show()
    # print(img.shape)

