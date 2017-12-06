#!/usr/bin/env python

import numpy as np
import argparse
import scipy
from scipy import sparse
from numpy import sqrt
import matplotlib
from matplotlib import pyplot as plt
import math
from math import pow
import pywt

def D4filter_bank():
    dec_lo = np.array([[0.7071067811865476, 0.7071067811865476]])
    dec_lo = dec_lo.T
    dec_hi = np.copy(dec_lo[::-1])
    for i in range(np.size(dec_hi)):
        dec_hi[i] = dec_hi[i]*pow(-1,i+1)
    rec_lo = np.copy(dec_lo[::-1])
    rec_hi = np.copy(dec_hi[::-1])

    return [dec_lo, dec_hi, rec_lo, rec_hi]

def convolution_mat(filter, matdim):
    repeat_rows = filter*np.ones((1,matdim))
    filter_mat = sparse.spdiags(repeat_rows, np.arange(D4scal.size), matdim, matdim).tocsr()
    filter_trans = filter_mat.T
    return (filter_mat, filter_trans)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Filters possion noise from 1D signal applying algorithm using Daubechies Wavelet')
    parser.add_argument('inpath', type = str, help = 'Inpath to RAW file')
    args = parser.parse_args()
    imgs = np.memmap(args.inpath, dtype = 'uint16', mode = 'r', shape = (15000, 4, 512, 512), order ='C')
    img = imgs[1,1,:,:]
    filtered_img = np.zeros((512,512))
    
    for row in range(512):
        img_row = img[row,:]
        sz = np.size(img_row)

        row_diag = sparse.spdiags(img_row, 0, sz, sz).tocsr()

        #   Pointwise square D4 scaling coeffs

        D4scal, D4wave, rD4scal, rD4wave = D4filter_bank()
        (D4scal_mat, D4scal_trans) = convolution_mat(D4scal, sz)
        (D4wave_mat, D4wave_trans) = convolution_mat(D4wave, sz)

        sqbasis = []
        pad = 0
        hdown2 = row_diag
        for i in range(9):
            hmapping = D4scal_trans*hdown2*D4scal_mat
            gmapping = D4wave_trans*hdown2*D4wave_mat
            hdown2 = hmapping[::2, ::2]
            gdown2 = gmapping[::2, ::2]
            hscale = hdown2.diagonal()
            gscale = gdown2.diagonal()
            D4scal_mat = D4scal_mat[0:len(hscale), 0:len(hscale)]
            D4scal_trans = D4scal_trans[0:len(hscale), 0:len(hscale)]
            D4wave_mat = D4wave_mat[0:len(hscale), 0:len(hscale)]
            D4wave_trans = D4wave_trans[0:len(hscale), 0:len(hscale)]
            sqbasis.append(gscale)
            if len(hscale) == 1:
                sqbasis.append(hscale)
            pad = pad + len(hscale)
        sqbasis = sqbasis[::-1]

        # Row Wavelet Transform

        coeffs = pywt.wavedec(img_row, 'db1')
    
        for i in range(0,len(coeffs)):
            coef = coeffs[i]
            filt = sqbasis[i]
            h = np.divide(np.multiply(coef,coef)-filt, np.multiply(coef, coef))
            h[h < 0] = 0
            h[h == np.nan] = 0
            coef = np.multiply(coef,h)
            coeffs[i] = coef

        # Perform Reconstruction

        filtered_signal = pywt.waverec(coeffs, 'db1')
        filtered_img[row,:] = filtered_signal
        
        # Graph Signals

#        pts = np.linspace(0,511,512)
#        f, (ax1,ax2) = plt.subplots(2, sharey=True)
#        ax1.plot(pts,img_row)
#        ax2.plot(pts, filtered_signal)
#        plt.show()

    for col in range(512):
        img_col = img[:,col]
        sz = np.size(img_row)

        col_diag = sparse.spdiags(img_row, 0, sz, sz).tocsr()

        #   Pointwise square D4 scaling coeffs

        D4scal, D4wave, rD4scal, rD4wave = D4filter_bank()
        (D4scal_mat, D4scal_trans) = convolution_mat(D4scal, sz)
        (D4wave_mat, D4wave_trans) = convolution_mat(D4wave, sz)

        sqbasis = []
        pad = 0
        hdown2 = col_diag
        for i in range(9):
            hmapping = D4scal_trans*hdown2*D4scal_mat
            gmapping = D4wave_trans*hdown2*D4wave_mat
            hdown2 = hmapping[::2, ::2]
            gdown2 = gmapping[::2, ::2]
            hscale = hdown2.diagonal()
            gscale = gdown2.diagonal()
            D4scal_mat = D4scal_mat[0:len(hscale), 0:len(hscale)]
            D4scal_trans = D4scal_trans[0:len(hscale), 0:len(hscale)]
            D4wave_mat = D4wave_mat[0:len(hscale), 0:len(hscale)]
            D4wave_trans = D4wave_trans[0:len(hscale), 0:len(hscale)]
            sqbasis.append(gscale)
            if len(hscale) == 1:
                sqbasis.append(hscale)
            pad = pad + len(hscale)
        sqbasis = sqbasis[::-1]

        # Row Wavelet Transform and filter

        coeffs = pywt.wavedec(img_col, 'db1')
    
        for i in range(1,len(coeffs)):
            coef = coeffs[i]
            filt = sqbasis[i]
            h = np.divide(np.multiply(coef,coef)-filt, np.multiply(coef, coef))
            h[h < 0] = 0
            h[h == np.nan] = 0
            coef = np.multiply(coef,h)
            coeffs[i] = coef

        # Perform Reconstruction

        filtered_signal = pywt.waverec(coeffs, 'db1')
        filtered_img[:,col] = filtered_signal
        
    # Plot imgs

    plt.imshow(filtered_img)
    plt.show()
    plt.imshow(img)
    plt.show()
