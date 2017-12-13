#!/usr/bin/env python

import numpy as np
from libtiff import TIFF
import matplotlib
from matplotlib import pyplot as plt
import skimage
from skimage import img_as_uint
from skimage import img_as_float
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Take raw stack and apply average across time')
    parser.add_argument('inpath', type = str, help = 'Inpath to raw stack')
    parser.add_argument('outpath', type = str, help = 'Outpath tiff stack')
    parser.add_argument('--size', type = int, default = 100, help = 'Number of images to average over')
    parser.add_argument('--startframe', type = int, default = 1, help = 'First frame to apply filter')
    parser.add_argument('--stopframe', type = int, default = 100, help = 'Last frame to apply filter')

    args = parser.parse_args()

    imgs = np.memmap(args.inpath, dtype = 'float64', mode = 'r', shape = (500, 512, 512), order ='C')

    tiff = TIFF.open(args.outpath, mode='w')

    avimg = np.zeros((512,512))

    for i in range(args.stopframe):
        for j in range(args.size):
            avimg = avimg + imgs[i+j,:,:]
            
        avimg = avimg/args.size
        tiff.write_image(img_as_uint(avimg))
