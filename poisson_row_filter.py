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
    dec_lo = np.array([[-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523]])
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

def adjustsz(matrix, declevel):
    for i in range(declevel):
        matrix = matrix[0:len(matrix.diagonal())/2, 0:len(matrix.diagonal())/2]
    return matrix

def D4waverec(cA, cD, lo_pass, hi_pass):
    cAup = np.zeros((2*len(cA)))
    cDup = np.zeros((2*len(cD)))
    cAup[0:2*len(cA):2] = cA
    cDup[0:2*len(cD):2] = cD
    Lo_R = lo_pass*cAup
    Hi_R = hi_pass*cDup
    signal = Lo_R + Hi_R
    return signal  

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Filters possion noise from 1D signal applying algorithm using Daubechies Wavelet')
    parser.add_argument('inpath', type = str, help = 'Inpath to RAW file')

    args = parser.parse_args()
    imgs = np.memmap(args.inpath, dtype = 'uint16', mode = 'r', shape = (15000, 4, 512, 512), order ='C')
    img = imgs[1,1,:,:]
    img_row = img[1,:]
    sz = np.size(img_row)

    img_diag = sparse.spdiags(img_row, 0, sz, sz).tocsr()

#   Pointwise square D4 scaling coeffs

    D4scal, D4wave, rD4scal, rD4wave = D4filter_bank()
    (D4scal_mat, D4scal_trans) = convolution_mat(D4scal, sz)
    (D4wave_mat, D4wave_trans) = convolution_mat(D4wave, sz)

    grow = []
    pad = 0
    hdown2 = img_diag
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
        grow.append(gscale)
        if len(hscale) == 1:
            grow.append(hscale)
        pad = pad + len(hscale)
    grow = grow[::-1]

    # Row Wavelet Transform

    D4scal, D4wave, rD4scal, rD4wave = D4filter_bank()
    (D4scal_mat, D4scal_trans) = convolution_mat(D4scal, sz)
    (D4wave_mat, D4wave_trans) = convolution_mat(D4wave, sz)

    coeffs = []
    pad = 0
    sdown2 = img_row
    for i in range(9):
        mapping1 = D4scal_trans*sdown2
        mapping2 = D4wave_trans*sdown2
        sdown2 = mapping1[::2]
        wdown2 = mapping2[::2]
        D4scal_trans = D4scal_trans[0:len(sdown2), 0:len(sdown2)]
        D4wave_trans = D4wave_trans[0:len(sdown2), 0:len(sdown2)]
        coeffs.append(wdown2)
        if sdown2.size == 1:
            coeffs.append(sdown2)
        pad = pad + len(sdown2)
    coeffs = coeffs[::-1]
    plt.show()
    # Filter Coefficients

    for i in range(1,len(coeffs)):
        coef = coeffs[i]
        filt = grow[i]*0
        h = np.divide(np.multiply(coef,coef)-filt, np.multiply(coef, coef))
        h[h < 0] = 0
        h[h == np.nan] = 0
        coef = np.multiply(coef,h)
        coeffs[i] = coef

        
    # Perform Reconstruction

    (rD4scal_mat, rD4scal_trans) = convolution_mat(rD4scal, sz)
    (rD4wave_mat, rD4wave_trans) = convolution_mat(rD4wave, sz)

    filtered_signal = coeffs[0]
    for i in range(9):
        cD = coeffs[i+1]
        Recwave = adjustsz(rD4wave_trans, 8-i)
        Recscal = adjustsz(rD4scal_trans, 8-i)
        filtered_signal = D4waverec(filtered_signal, cD, Recscal, Recwave)

    # Graph Signals

    pts = np.linspace(0,511,512)
    f, (ax1,ax2) = plt.subplots(2, sharey=True)
    ax1.plot(pts,img_row)
    ax2.plot(pts, filtered_signal)
    plt.show()
