#!/usr/bin/env python

import numpy as np
import argparse
import pywt
from skimage import img_as_uint
import matplotlib
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)

def PoissonDWT2(image):

    '''Change for loop index to adjust filter levels'''
    
    imgsqrt = np.sqrt(image)
    coeffs = pywt.wavedec2(imgsqrt, 'db1')
    scales = len(coeffs)
    for i in range(scales-1, scales):
        (cH, cV, cD) = coeffs[i]
        conc1 = np.concatenate((cH,cV))
        conc2 = np.concatenate((conc1,cD))
        threshold = (np.sqrt(2*np.log(imgsqrt.size))*np.median(abs(conc2)))/0.6745
        cD[cD < threshold] = 0
        cH[cH < threshold] = 0
        cV[cV < threshold] = 0
        coeffs[i] = (cH, cV, cD)
    filtered_signal = pywt.waverec2(coeffs, 'db1')
    img = np.square(filtered_signal)
    img = img/float(np.max(img))
    img = img_as_uint(img)
    return img
