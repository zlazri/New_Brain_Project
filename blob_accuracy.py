#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import skimage
import argparse
from libtiff import TIFF3D
from skimage.feature import blob_log
from math import sqrt
from overlap import blob_overlap
import itertools as itt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Compares denoising methods to "Ground Truth" using LoG')
    parser.add_argument('groundstack', type = str, help = 'Path to stack of ground truth images')
    parser.add_argument('rankstack', type = str, help = 'Path to stack of rank filtered images')
    parser.add_argument('FFTrstack', type = str, help = 'Path to stack of FFTreg filtered images')
    parser.add_argument('--threshold', type = int, default=0.015, help = 'Threshold for the scale space maxima')
    parser.add_argument('--overlap', type = int, default=0.85, help = 'The amount of overlap required between a "ground truth" blob and "method" blob for them to be considered the same blob')

    args = parser.parse_args()

    groundtiff = TIFF3D.open(args.groundstack, mode = 'r')
    ranktiff = TIFF3D.open(args.rankstack, mode = 'r')
    FFTrtiff = TIFF3D.open(args.FFTrstack, mode = 'r')

    groundstack = groundtiff.read_image()
    rankstack = ranktiff.read_image()
    FFTrstack = FFTrtiff.read_image()

    assert(groundstack.shape == rankstack.shape == FFTrstack.shape)

    frames = groundstack.shape[0]

    rankratios = []
    FFTrratios = []

    for i in range(frames):
        groundblobs = blob_log(groundstack[i,:,:], max_sigma=11, min_sigma=4, num_sigma=14, threshold=args.threshold)
        rankblobs = blob_log(rankstack[i,:,:], max_sigma=11, min_sigma=4, num_sigma=14, threshold=args.threshold)
        FFTrblobs = blob_log(FFTrstack[i,:,:], max_sigma=13, min_sigma=4, num_sigma=14, threshold=args.threshold)

        groundblobs[:,2] = groundblobs[:,2] * sqrt(2)
        rankblobs[:,2] = rankblobs[:,2] * sqrt(2)
        FFTrblobs[:,2] = FFTrblobs[:,2] * sqrt(2)

        truerank=[]
        falserank=[]
        trueFFTr=[]
        falseFFTr=[]
                
        for groundblob in groundblobs:
            dum1 = 0
            dum2 = 0
            find1 = False
            find2 = False
            for rankblob in rankblobs:
                if blob_overlap(groundblob,rankblob) > args.overlap:
                    truerank.append(rankblob)
                    find1 = True
                else:
                    dum1 = dum1 + 1
                    if dum1 == len(rankblobs[:,0]):
                        falserank.append(rankblob)
                if find1:
                    break
                
            for FFTrblob in FFTrblobs:
                if blob_overlap(groundblob,FFTrblob) > args.overlap:
                    trueFFTr.append(FFTrblob)
                    find2 = True
                else:
                    dum2 = dum2 + 1
                    if dum2 == len(FFTrblobs[:,0]):
                        falseFFTr.append(FFTrblobs)
                if find2:
                    break
                
        rankratios.append(float(len(truerank))/float(len(groundblobs)))
        FFTrratios.append(float(len(trueFFTr))/float(len(groundblobs)))

        print('Calculated ratio ' + str(i))

    # Graph Results
        
    pts = np.linspace(0,99,100)
    f, (ax1,ax2) = plt.subplots(2, sharey= True, sharex=True)
    ax1.plot(pts,rankratios)
    ax1.set_title("Rank to Ground Truth Blob Percentages Across Frames")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Percentage Match")
    ax2.plot(pts,FFTrratios)
    ax2.set_title("FFTreg to Ground Truth Blob Percentages Across Frames")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Percentage Match")
    plt.show()

    # Boxplot results

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.boxplot(rankratios, 0, 'rs', 0)
    ax1.set_xlim([0,1.1])
    ax1.set_title("Rank to Ground Truth Blob Percentages")
    ax2.boxplot(FFTrratios, 0, 'rs', 0)
    ax2.set_xlim([0,1.1])
    ax2.set_title("FFTreg to Ground Truth Blob Percentages")
    ax2.set_xlabel("Percentage Match")
    plt.show()
        
                    
#        print(groundblobs)
#        print(rankblobs)
#        print(FFTrblobs)
        
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    size1 = np.shape(groundblobs)[0]
    size2 = np.shape(rankblobs)[0]
    size3 = np.shape(FFTrblobs)[0]

    for i in range(size1):
        y, x, r = groundblobs[i,:]
        c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
        ax1.add_patch(c)
    for i in range(size2):
        y, x, r = rankblobs[i,:]
        c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
        ax2.add_patch(c)
    for i in range(size3):
        y, x, r = FFTrblobs[i,:]
        c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
        ax3.add_patch(c)

    ax1.set_title("Ground Truth Blob LoG")
    ax2.set_title("Rank Filter Blob LoG")
    ax3.set_title("FFTreg Blob LoG")
    
    ax1.imshow(groundstack[0,:,:])
    ax2.imshow(rankstack[0,:,:])
    ax3.imshow(FFTrstack[0,:,:])
    plt.show()
