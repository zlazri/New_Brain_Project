import numpy as np

def gauss_filt2D(n, sig):

    ''' Creates a 2D gaussian kernel.
      
        n = size of gaussian kernel-- n x n
        sig = sigma
    '''

    nx = ny = np.linspace(-np.floor(n/2), np.floor(n/2), n)

    X, Y = np.meshgrid(nx, ny)

    kernel = np.exp(-(np.square(X) + np.square(Y)) / (2*sig*sig))/(2*np.pi*sig*sig)

    kernel = kernel/np.sum(kernel)

    return kernel
