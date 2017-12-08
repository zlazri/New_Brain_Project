#!/usr/bin/env python

import numpy as np
from libtiff import TIFF
import matplotlib
from matplotlib import pyplot as plt
import skimage
from skimage import img_as_uint
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Take raw stack and apply average across time')
    parser.add_argument('inpath', type = str, help = 'Inpath to raw stack')
    parser.add_argument('outpath', type = str, help = 'Outpath tiff stack')
    parser.add_argument('--size', type = int, default = 100, help = 'Number of frames to average images over')
    parser.add_argument('--startframe', type = int, default = 1, help = 'First frame to apply filter')
    parser.add_argument('--stopframe', type = int, default = 100, help = 'Last frame to apply filter')

    args = parser.parse_args()

    imgs = np.memmap(args.inpath, dtype = 'float64', mode = 'r', shape = (args.stopframe + args.size, 512, 512), order ='C')

    tiff = TIFF.open(args.outpath, mode='w')

    for i in range(args.stopframe):
        for j in range(args.size):
            if j == 0:
                avimg = imgs[i+j,:,:]
            else:
                avimg = avimg + imgs[i+j,:,:]

        avimg = avimg/args.size
        avimg = img_as_uint(avimg)

#        plt.imshow(avimg)
#        plt.show()

        #assert(1==0)

        tiff.write_image(avimg)
