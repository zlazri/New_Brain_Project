#!/usr/bin/env python

import numpy as np
from skimage import img_as_uint
import argparse
from libtiff import TIFF
from matplotlib import pyplot as plt

def rank_filter2D(img, sz, rank):

    ''' This function filters an image with a 2D mean filter

        img = input image
        r = size parameter (look at the actual code to see how to use it)
    '''

    m1, m2 = img.shape
    I = sz/2
    outimg = np.zeros((m1,m2), dtype=float)
    
    for i in range(sz, m1-sz-1):
        for j in range(sz, m2-sz-1):
            sortarr = np.sort(img[i-I:i+I+1,j-I:j+I+1], axis = None)
            outimg[i,j] = sortarr[rank-1]
            
    for i in range(sz) + range(m1-sz-1, m1):
        for i in range(sz) + range(m2-sz-1, m2):
            outimg[i,j] = img[i,j]

    outimg = outimg/float(np.max(outimg))
    outimg = img_as_uint(outimg)

    return outimg

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Performs rank filtering to image')
    parser.add_argument('inpath', type=str, help = 'Inpath to tiff stack')
    parser.add_argument('outpath', type=str, help = 'Outpath for filtered images')
    parser.add_argument('--sz', type=int, default=5, help = 'Size of the rank filter neighborhood is 2*sz+1 by 2*sz+1')
    parser.add_argument('--rank', type=int, default=5, help = 'nth largest element of the neighborhood')
    parser.add_argument('--frames', type=int, default=100,  help = 'Number of frames to which the filter is applied')
    args = parser.parse_args()

    allimgs = np.memmap(args.inpath, dtype = 'uint16', mode = 'r', shape = (15000, 4, 512, 512), order ='C')
    imgs = allimgs[:,1,:,:]
    tiffout = TIFF.open(args.outpath, mode='w')
    for i in range(args.frames):
#        tiffout.write_image(rank_filter2D(imgs[i,:,:], args.sz, args.rank))
        plt.imshow(rank_filter2D(imgs[i,:,:], args.sz, args.rank))
        plt.show()
