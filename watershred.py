# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:39:01 2018

@author: liu
"""
import numpy as np
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from analysis import analysis
from skimage.morphology import watershed, h_maxima
from scipy.ndimage import distance_transform_edt, binary_dilation, label, generate_binary_structure
from skimage.io import imread, imsave

if __name__ == "__main__":
    img = (imread('data/cell.png', 1)*255).astype('uint8')
    _, _, _, labeled1 = analysis(img)

    fig, axes = plt.subplots(2, 3)
    ax = axes.flatten()

    ls = regionprops(labeled1)
    lst = np.array([0]*(len(ls)+1))  

    # eccentricity filter             
    for i in ls:
        if i.eccentricity<0.7 : lst[i.label] = 0
        else : lst[i.label] = i.label
    img_filter = lst[labeled1.astype(np.int32)]

    # relabel
    labeled2 = label(img_filter.copy(), generate_binary_structure(2, 2))[0].astype('uint8')

    # distance transform
    lst = [i for i in range(256)]
    lst = np.array(lst[::-1])
    distance = distance_transform_edt(labeled2)

    #find the maximum as the seed
    markers = h_maxima(distance, 2)
    maximums = np.where(markers>0)
    print(maximums)
    markers = binary_dilation(markers, structure=generate_binary_structure(2, 2))
    img_markers = label(markers, generate_binary_structure(2, 2))[0].astype('uint8')
    labeled1 = lst[labeled1.astype(np.int32)]   

    #分水岭
    watershed = watershed(labeled1, img_markers, watershed_line=True)
    lst = np.array([1]+[0]*254)
    #提取出边界
    watershed_line = lst[watershed]

    lst=np.array([0]+[1]*254)
    img1=lst[img_filter]

    ax[0].imshow(labeled1)
    ax[0].set_title('labeld', fontsize=12)

    ax[1].imshow(labeled2)
    ax[1].set_title('img_filter', fontsize=12)

    ax[2].imshow(distance)
    ax[2].set_title('distance', fontsize=12)

    ax[3].imshow(distance)
    ax[3].scatter(maximums[1], maximums[0], c='r', s=20)
    ax[3].set_title('find maximum', fontsize=12)

    # ax[4].imshow(distance)
    ax[4].imshow(distance+watershed_line*10)
    # ax[4].scatter(maximums[1], maximums[0], c='r', s=20)
    ax[4].set_title('watershed line', fontsize=12)   

    # ax[4].imshow(distance)
    line_dialation = binary_dilation(img1*watershed_line, structure=generate_binary_structure(2, 2))
    ax[5].imshow(line_dialation+img1)
    # ax[4].scatter(maximums[1], maximums[0], c='r', s=20)
    ax[5].set_title('split cells', fontsize=12)  

    for i in range(6): ax[i].set_axis_off()
    fig.tight_layout()
    plt.show()




    