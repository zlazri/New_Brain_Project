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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Filters possion noise from 1D signal applying algorithm using Daubechies Wavelet')
    parser.add_argument('inpath', type = str, help = 'Inpath to RAW file')

    args = parser.parse_args()
    imgs = np.memmap(args.inpath, dtype = 'uint16', mode = 'r', shape = (15000, 4, 512, 512), order ='C')
    img = imgs[1,1,:,:]
    img_row = img[1,:]
    sz = np.size(img_row)

    img_diag = sparse.spdiags(img_row, 0, sz, sz).tocsr()

#   pointwise square D4 scaling coeffs

    D4scal = np.array([[1+sqrt(3), 3+sqrt(3), 3-sqrt(3), 1-sqrt(3)]])/4
    D4scal = D4scal.T
    repeat_rows = D4scal*np.ones((1,sz))

    D4scal_mat = sparse.spdiags(repeat_rows, np.array([0, 1, 2, 3]), sz, sz).tocsr()
    D4scal_trans = D4scal_mat.T

    hmapping = D4scal_trans*img_diag*D4scal_mat
    hdown2 = hmapping[::2, ::2]
    hrow = hdown2.diagonal()

#   Pointwise Square D4 detail coeffs

    D4wave = D4scal[::-1]
    for i in range(np.size(D4scal)):
        D4wave[i] = D4wave[i]*pow(-1,i)

    D4wave = D4wave
    repeat_rows = D4wave*np.ones((1,sz))

    D4wave_mat = sparse.spdiags(repeat_rows, np.array([0, 1, 2, 3]), sz, sz).tocsr()
    D4wave_trans = D4wave_mat.T

    gmapping = D4wave_trans*img_diag*D4wave_mat
    gdown2 = gmapping[::2, ::2]
    grow = gdown2.diagonal()

#   Row Wavelet Transform

    coeffs = pywt.dwt(img_row.T, 'db1')
    
    (cS, cD) = coeffs

#    print(cS.size)

    h1 = np.divide(np.multiply(cS,cS)-hrow, np.multiply(cS, cS))
    h1[h1 < 0] = 0
    h1[h1 == np.nan] = 0

#    print(h1)

    h2 = np.divide(np.multiply(cD,cD)-grow, np.multiply(cD, cD))
    h2[h2 < 0] = 0
    h2[h2 == np.nan] = 0
    
    cS = np.multiply(cS,h1)

    cD = np.multiply(cD,h2)

    filtered_signal = pywt.idwt(cS, cD, 'db1')

    print(filtered_signal)
    print('----------')
    print(img_row)
    
    pts = np.linspace(0,511,512)
    f, (ax1,ax2) = plt.subplots(2, sharey=True)
    ax1.plot(pts,img_row)
    ax2.plot(pts, filtered_signal)
    plt.show()
