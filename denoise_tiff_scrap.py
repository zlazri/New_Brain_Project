#!/usr/bin/env python

import numpy as np
import argparse
from libtiff import TIFF
import pywt
import skimage
from skimage import img_as_uint
import matplotlib
from matplotlib import pyplot as plt
import denoise_scrap
from denoise_scrap import PoissonDWT2
import matplotlib
from matplotlib import pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Take tiff stack and apply denoising')
    parser.add_argument('inpath', type = str, help = 'Inpath to tiff stack')

    args = parser.parse_args() 
    
    tiff = TIFF.open(args.inpath, mode='r')
    for image in tiff.iter_images():
        PoissonDWT2(image)
