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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Filters possion noise from image applying algorithm using Daubechies Wavelet')
    parser.add_argument('inpath', type = str, help = 'Inpath to RAW file')

    args = parser.parse_args()
    imgs = np.memmap(args.inpath, dtype = 'uint16', mode = 'r', shape = (15000, 4, 512, 512), order ='C')
    img = imgs[1,1,:,:]
    sz = np.size(img[:,1])

    # D4scal and D4scal'

    D4scal, D4wave, rD4scal, rD4wave = D4filter_bank()
    repeat_rows = D4scal*np.ones((1,sz))

    D4scal_mat = sparse.spdiags(repeat_rows, np.array([0, 1, 2, 3, 4, 5, 6, 7]), sz, sz).tocsr()
    D4scal_trans = D4scal_mat.T

    # D4wave and D4wave'

    repeat_rows = D4wave*np.ones((1,sz))

    D4wave_mat = sparse.spdiags(repeat_rows, np.array([0, 1, 2, 3, 4, 5, 6, 7]), sz, sz).tocsr()
    D4wave_trans = D4wave_mat.T

    # LL, LH, HL, HH

    LL = img*D4scal_mat*D4scal_mat
    LL = LL[:, ::2]
    LL = D4scal_trans*D4scal_trans*LL
    LL = LL[::2, :]

    LH = img*D4scal_mat*D4scal_mat
    LH = LH[:, ::2]
    LH = D4wave_trans*D4wave_trans*LH
    LH = LH[::2, :]

    HL = img*D4wave_mat*D4wave_mat
    HL = HL[:, ::2]
    HL = D4scal_trans*D4scal_trans*HL
    HL = HL[::2, :]

    HH = img*D4wave_mat*D4wave_mat
    HH = HH[:, ::2]
    HH = D4wave_trans*D4wave_trans*HH
    HH = HH[::2, :]

    # Image Decompositon
    
    imgcA = img*D4scal_mat
    imgcA = imgcA[:, ::2]
    imgcA = D4scal_trans*imgcA
    imgcA = imgcA[::2, :]
        
    imgcH = img*D4scal_mat
    imgcH = imgcH[:, ::2]
    imgcH = D4wave_trans*imgcH
    imgcH = imgcH[::2, :]

    imgcV = img*D4wave_mat
    imgcV = imgcV[:, ::2]
    imgcV = D4scal_trans*imgcV
    imgcV = imgcV[::2, :]
    
    imgcD = img*D4wave_mat
    imgcD = imgcD[:, ::2]
    imgcD = D4wave_trans*imgcD
    imgcD = imgcD[::2, :]

    ll = np.divide(np.multiply(imgcA,imgcA)-LL, np.multiply(imgcA, imgcA))
    ll[ll < 0] = 0
    ll[ll == np.nan] = 0
    
    lh = np.divide(np.multiply(imgcH,imgcH)-LH, np.multiply(imgcH, imgcH))
    lh[lh < 0] = 0
    lh[lh == np.nan] = 0

    hl = np.divide(np.multiply(imgcV,imgcV)-HL, np.multiply(imgcV, imgcV))
    hl[hl < 0] = 0
    hl[hl == np.nan] = 0

    hh = np.divide(np.multiply(imgcD,imgcD)-HH, np.multiply(imgcD, imgcD))
    hh[hh < 0] = 0
    hh[hh == np.nan] = 0

    cA = imgcA
    cH = np.multiply(imgcH,lh)
    cV = np.multiply(imgcV,hl)
    cD = np.multiply(imgcD,hh)

    # Image Reconstruction
    
    mdim = ndim = len(img)

    repeat_rows = rD4scal*np.ones((1,sz))
    rD4scal_mat = sparse.spdiags(repeat_rows, np.array([0, 1, 2, 3, 4, 5, 6, 7]), sz, sz).tocsr()
    rD4scal_trans = rD4scal_mat.T

    repeat_rows = rD4wave*np.ones((1,sz))
    rD4wave_mat = sparse.spdiags(repeat_rows, np.array([0, 1, 2, 3, 4, 5, 6, 7]), sz, sz).tocsr()
    rD4wave_trans = rD4wave_mat.T
    
    # Upsample columns
    cAup = np.zeros((mdim,ndim/2))
    cAup[0:mdim:2, :] = cA
    
    cHup = np.zeros((mdim, ndim/2))
    cHup[0:mdim:2, :] = cH

    cVup = np.zeros((mdim, ndim/2))
    cVup[0:mdim:2, :] = cV

    cDup = np.zeros((mdim, ndim/2))
    cDup[0:mdim:2, :] = cD

    # Convolve columns and add

    cAup = rD4scal_trans*cAup
    cHup = rD4scal_trans*cHup
    cVup = rD4wave_trans*cVup
    cDup = rD4wave_trans*cDup

    Lo_R = cAup + cHup
    Hi_R = cVup + cDup

    # Upsample rows

    Lo_up = np.zeros((mdim,ndim))
    Lo_up[:, 0:ndim:2] = Lo_R

    Hi_up = np.zeros((mdim,ndim))
    Hi_up[:, 0:ndim:2] = Hi_R

    # Convole rows and add

    Lo_up = Lo_up*rD4scal_mat
    Hi_up = Hi_up*rD4wave_mat

    newimg = Lo_up + Hi_up
        
    filtered_signal = pywt.idwt2((cA,(cH, cV, cD)), 'db4')

    plt.imshow(newimg)
    plt.show()
