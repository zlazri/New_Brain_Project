#!/usr/bin/env python

import numpy as np
from skimage import img_as_uint
import argparse
from libtiff import TIFF
from matplotlib import pyplot as plt

def rank_filter2D(img, sz, rank):

    ''' This function filters an image with a 2D mean filter

        img = input image
        sz = size parameter (look at the actual code to see how to use it)
        rank = the position of the number you select from a neighborhood

    '''
    
    m1, m2 = img.shape
    I = sz/2
    outimg = np.zeros((m1,m2), dtype='uint16')
    
    for i in range(sz, m1-sz-1):
        for j in range(sz, m2-sz-1):
            sortarr = np.sort(img[i-I:i+I+1,j-I:j+I+1], axis = None)
            outimg[i,j] = sortarr[rank-1]
            
    return outimg

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Performs rank filtering to image')
    parser.add_argument('inpath', type=str, help = 'Inpath to raw stack')
    parser.add_argument('outpath', type=str, help = 'Outpath for filtered images')
    parser.add_argument('--sz', type=int, default=5, help = 'Size of the rank filter neighborhood is NxN')
    parser.add_argument('--rank', type=int, default=5, help = 'Nth largest element of the neighborhood')
    parser.add_argument('--frames', type=int, default=100,  help = 'Number of frames to which the filter is applied')
    args = parser.parse_args()

    imgs = np.memmap(args.inpath, dtype = 'uint16', mode = 'r', shape = (args.frames, 4, 512, 512), order ='C')
    imgs = imgs[:,1,:,:]
    
    tiffout = TIFF.open(args.outpath, mode='w')
    for i in range(args.frames):
        tiffout.write_image(rank_filter2D(imgs[i,:,:], args.sz, args.rank))
        print("filtered image: " + str(i))
