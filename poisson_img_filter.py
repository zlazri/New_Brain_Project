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
import sparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Filters possion noise from image applying algorithm using Daubechies Wavelet')
    parser.add_argument('inpath', type = str, help = 'Inpath to RAW file')

    args = parser.parse_args()
    imgs = np.memmap(args.inpath, dtype = 'uint16', mode = 'r', shape = (15000, 4, 512, 512), order ='C')
    img = imgs[1,1,:,:]
    sz = np.size(img)


    # Create 4D image array
    
    data = img.ravel()
    coors = np.zeros((4, sz))
    b = 0
    for r in range(512):
        for s in range(512):
            if r == s:
                for m in range(512):
                    for n in range(512):
                        if m == n:
                            coors[:,b] = [r,s,m,n]
                            b = b+1
    n = 512
    ndims = 4
    nnz = sz
    coors = coors.astype(int)
    diag4D = sparse.COO(coors, data, shape=((n,) * ndims))

    # Create hhhh filter

    D4scal = np.array([[1+sqrt(3), 3+sqrt(3), 3-sqrt(3), 1-sqrt(3)]])/4
    D4scal = D4scal.T
    repeat_rows = D4scal*np.ones((1,sz))

    D4scal_mat = sparse.spdiags(repeat_rows, np.array([0, 1, 2, 3]), sz, sz).tocsr()    
    h1_coors = 

    
    print(diag4D)
