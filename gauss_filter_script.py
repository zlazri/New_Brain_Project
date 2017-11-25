#!/usr/bin/env python

import numpy as np
import gaussian_filter2D
from gaussian_filter2D import gauss_kern2D
from gaussian_filter2D import gauss_filt2D
import argparse
from libtiff import TIFF
from scipy.ndimage import filters
from scipy.ndimage.filters import gaussian_filter
import matplotlib
from matplotlib import pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Extracts imgs from RAW file, filters with Gaussian kernel, then writes to a tiff stack')
    parser.add_argument('inpath', type = str, help = 'Inpath to RAW file')
    parser.add_argument('outpath', type = str, help = 'Outpath to tiff file')
    parser.add_argument('--startframe', type = int, default = 1, help = 'First frame from the RAW path to extract')
    parser.add_argument('--stopframe', type = int, default = 100, help = 'Last frame from the RAW path to extract')

    args = parser.parse_args()
    tiff = TIFF.open(args.outpath, mode='w')
    imgs = np.memmap(args.inpath, dtype = 'uint16', mode = 'r', shape = (15000, 4, 512, 512), order ='C')
    
    for i in range(args.startframe, args.stopframe):
        tiff.write_image(gaussian_filter(imgs[i,1,:,:], 1, truncate=1))
        print('Filtered and stored image: ', i)
        
