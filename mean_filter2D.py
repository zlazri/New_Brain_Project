import numpy as np
from skimage import img_as_uint

def mean_filt2D(img, r):

    ''' This function filters an image with a 2D mean filter

        img = input image
        r = size parameter (look at the actual code to see how to use it)
    '''

    m1, m2 = img.shape
    I = np.linspace(-r,r,2*r+1)
    I = I.astype(int)
    outimg = np.zeros((m1,m2), dtype=float)

    for i in range(r, m1-r-1):
        for j in range(r, m2-r-1):
            outimg[i,j] = np.mean(img[i+I,j+I])
            
    for i in range(r) + range(m1-r-1, m1):
        for i in range(r) + range(m2-r-1, m2):
            outimg[i,j] = img[i,j]

    outimg = outimg/float(np.max(outimg))
    outimg = img_as_uint(outimg)

    return outimg
