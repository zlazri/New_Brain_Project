import numpy as np

def gauss_kern1D(n, sig):

    ''' Creates a 1D gaussian kernel.
      
        n = size of gaussian kernel
        sig = sigma
    '''

    

def gauss_kern2D(n, sig):

    ''' Creates a 2D gaussian kernel.
      
        n = size of gaussian kernel-- n x n
        sig = sigma
    '''

    nx = ny = np.linspace(-np.floor(n/2), np.floor(n/2), n)

    X, Y = np.meshgrid(nx, ny)

    kernel = np.exp(-(np.square(X) + np.square(Y)) / (2*sig*sig))/(2*np.pi*sig*sig)

    kernel = kernel/np.sum(kernel)

    return kernel

def gauss_filt2D(img, Gfilt):

    ''' This function performs the 2D convolution of an image with a Gaussian kernel
    
    img = image that must be filtered
    Gfilt = Gaussian kernel

    Note: m1 and m2, below, should be the same since images are square-- at least for our purposes.

    '''

    m1, m2 = img.shape
    n1, n2 = Gfilt.shape    
    r = int(n1/2)
#    print(r)
    outimg = np.zeros((m1,m2), dtype=float)
    I = np.linspace(-r,r,n1)
    I = I.astype(int)
#    print(I)
    i0 = r+1
    center = (i0, i0)
    for i in range(r, m1-r-1):
        for j in range(r, m2-r-1):
            outimg[i,j] = np.sum(np.multiply(Gfilt,img[i-I, j-I]))
#            print(outimg[i,j])
#            assert(j<5)

#    for i in range(r) + range(m1-r-1, m1):
#        for j in range(r) + range(m2-r-1, m2):
#            outimg[i,j] = img[i,j]

    return outimg
