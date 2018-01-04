#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from libtiff import TIFF
import argparse
from math import log10

def PSNR(noisyimgs, imgs):
    
    ''' Takes a set of noisy (tiff) and denoised (raw) images and outputs the PSNR values for each image of the stack'''
    
    tiffims = TIFF.open(imgs, mode = 'r')
    PSNRvals = np.zeros(100)
    dum = 0
    for image in tiffims.iter_images():
        diff = image[7:505, 7:505]-noisyimgs[dum,:,:]
        sq = np.multiply(diff, diff)
        MSE = np.sum(sq)/(len(image[:,0])*len(image[0,:]))
        MAX = np.amax(image)
        PSNR = 20*log10(MAX)-10*log10(MSE)
        PSNRvals[dum] = PSNR
        dum = dum + 1
    return PSNRvals

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Perform PSNR on image')
    parser.add_argument('rankstack', type = str, help = 'Inpath to stack of rank images')
    parser.add_argument('groundstack', type = str, help = 'Inpath to stack of ground truth images')
    parser.add_argument('FFTrstack', type = str, help = 'Inpath to stack fo FFTreg images')
    parser.add_argument('noisestack', type = str, help = 'Inpath to stack of noisy images')
    parser.add_argument('--num', type = int, default = 100, help='number of frames on which to perform PSNR')

    # import images

    args = parser.parse_args()
    
    noisedata = np.memmap(args.noisestack, dtype = 'uint16', mode = 'r', shape = (15000, 4, 512, 512), order ='C')
    noisyimgs = noisedata[0:100, 1, 7:505, 7:505]

    PSNRrank = PSNR(noisyimgs, args.rankstack)    
    PSNRground = PSNR(noisyimgs, args.groundstack)
    PSNRreg = PSNR(noisyimgs, args.FFTrstack)
    pts = np.linspace(0, 99, 100)

    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    ax1.plot(pts, PSNRrank)
    ax1.set_title('Rank PSNR Values for 100 Frames')
    ax1.set_ylim([35, 45])
    ax2.plot(pts, PSNRground)
    ax2.set_title('Ground truth PSNR Values for 100 Frames')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_ylim([35, 45])
    ax3.plot(pts, PSNRreg)
    ax3.set_title('FFTreg PSNR Values for 100 Frames')
    ax3.set_xlabel('Frames')
    ax3.set_ylim([35, 45])
    plt.show()
