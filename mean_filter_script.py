#!/usr/bin/env python

import numpy as np
import mean_filter2D
from mean_filter2D import mean_filt2D
import argparse
from libtiff import TIFF

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Extracts imgs from RAW file, filters with mean filter, then writes to a tiff stack')
    parser.add_argument('inpath', type = str, help = 'Inpath to RAW file')
    parser.add_argument('outpath', type = str, help = 'Outpath to tiff file')
    parser.add_argument('--startframe', type = int, default = 1, help = 'First frame from the RAW path to extract')
    parser.add_argument('--stopframe', type = int, default = 100, help = 'Last frame from the RAW path to extract')

    args = parser.parse_args()
    tiff = TIFF.open(args.outpath, mode='w')
    imgs = np.memmap(args.inpath, dtype = 'uint16', mode = 'r', shape = (15000, 4, 512, 512), order ='C')

    for i in range(args.startframe, args.stopframe):
        tiff.write_image(mean_filt2D(imgs[i,1,:,:], 1))
        print('Filtered and stored image: ', i)
