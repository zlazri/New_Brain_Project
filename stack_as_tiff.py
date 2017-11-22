#!/usr/bin/env python

import argparse
import numpy as np
from libtiff import TIFF

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Take stack of RAWs and output tiff')
    parser.add_argument('inpath', type=str, help = 'Inpath to the RAW images')
    parser.add_argument('outpath', type=str, help = 'Outpath for the tiff file')
    parser.add_argument('--startframe', type=int, default=1, help = 'First frame to stack in the tiff file from the RAW file')
    parser.add_argument('--stopframe', type=int, default=100, help = 'Last frame to stack in the tiff file from the RAW file')

    args = parser.parse_args()
    tiff = TIFF.open(args.outpath, mode= 'w')
    for i in range(args.startframe, args.stopframe):
        img_chans = np.memmap(args.inpath, dtype = 'uint16', mode = 'r', shape = (i, 4, 512, 512), order ='C')
        img = img_chans[0,1,:,:]
        tiff.write_image(img)
    
