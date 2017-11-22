#!/usr/bin/env python

import numpy as np
import argparse
from libtiff import TIFF
import pywt
import skimage
from skimage import img_as_uint
import matplotlib
from matplotlib import pyplot as plt
import denoise
from denoise import PoissonDWT2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Take tiff stack and apply denoising')
    parser.add_argument('inpath', type = str, help = 'Inpath to tiff stack')
    parser.add_argument('outpath', type = str, help = 'Outpath for filtered images')
    args = parser.parse_args() 
    
    tiff = TIFF.open(args.inpath, mode='r')
    tiff2 = TIFF.open(args.outpath, mode='w')
    for image in tiff.iter_images():
        tiff2.write_image(PoissonDWT2(image))
