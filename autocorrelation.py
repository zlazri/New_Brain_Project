import numpy as np
from scipy.linalg import toeplitz

def AutoCorr(data, lag):
    assert(type(data).__name__ == 'ndarray')
    data = data.astype(float)
    
    length = len(data)
    mu = sum(data)/length
    denom = []

    for i in range(length):
        denom.append(np.square(data[i]-mu))
    denom = sum(denom)
    
    autocorr = []
    for i in range(lag, length):
        autocorr.append((data[i]-mu)*(data[i-lag]-mu))
    autocorr = np.array(autocorr)

    return autocorr

def AutoCorrMat(data, lag):
    autocorr = AutoCorr(data, lag)
    matrix = toeplitz(autocorr, autocorr)
    return matrix

def SE(data, lag):
    N = len(data)
    p = AutoCorr(data, lag)
    p_sq = np.dot(p, p)
    SE = np.sqrt((1+2*p_sq)/N)
    
    return SE

def Cov(data, lag=1):
    assert(type(data).__name__ == 'ndarray')
    data = data.astype(float)

    length = len(data)
    data1 = data[0:length-lag]
    data2 = data[lag:length]
    mu = sum(data)/length
    covariance = (1/length)*np.multiply(data1-mu, data2-mu)
        
    return covariance

def CovMATLAB(x, M):

    ''' x: The sequence for which we want to create an autocorrelation matrix.
        M: The length of each autocorrelation window

        NOTE: This function creates a matrix of 3 regions. The top region is the 
              'prewindow'. The middle region is the covariance matrix. The end
              region is the 'post window'. See MATLAB's corrmtx function 
              documentation for understanding.
    '''
    
    assert(type(x).__name__ == 'ndarray')
    x = x.astype(float)
    N = len(x)

    # Form middle region of matrix.
    Xtemp= arr_buffer(x, N-M, N-M-1, "nodelay")
    X_unscaled = Xtemp[:, ::-1]
    X = X_unscaled/np.sqrt(N-M)

    # Add prewindow to top region of matrix.
    Xtemp_u = arr_buffer(x[0:M], M, M-1)
    X_u = np.zeros((M,M+1))
    X_u[:,:-1] = Xtemp_u[:, ::-1]
    X_unscaled = np.append(X_u, X_unscaled, 0)

    # Add postwindow to lower region of matrix.
    Xtemp_l = arr_buffer(x[N-1:N-M-1:-1], M, M-1)
    X_l = np.zeros((M, M+1))
    X_l[:,1:] = Xtemp_l[::-1, :]
    X = np.append(X_unscaled, X_l, 0)/np.sqrt(N-1)

    return X

def arr_buffer(arr, n, p, opt="delay"):

    '''arr: The array from which we are going to form a buffer.
       n: The number of partitions in the buffer
       p: specified amount of overlap
    '''
    arr_sz = len(arr)
    mdim = n

    if opt == "delay":
        ndim = int(np.ceil(arr_sz/(n-p)))
                       
        leftover = (n-p)*ndim-arr_sz

        for i in range(leftover):
            arr = np.append(arr, 0)

        output = np.zeros((mdim, ndim))

        for i in range(ndim):
            output[p:n, i] = arr[i*(n-p):i*(n-p)+n-p]

        for i in range(1,ndim):
            output[0:p, i] = output[n-p:n, i-1]

    elif opt == "nodelay":
        ndim = int(np.ceil((arr_sz-p)/(n-p)))

        leftover = (n-p)*(ndim-1)+ n-arr_sz

        for i in range(leftover):
            arr = np.append(arr, 0)

        output = np.zeros((mdim,ndim))
        output[0:n, 0] = arr[0:n]

        for i in range(1, ndim):
            output[p:n, i] = arr[i*(n-p)+p:i*(n-p)+n]

        for i in range(1, ndim):
            output[0:p, i] = output[n-p:n, i-1]        

    else:
        raise AssertionError("unknown value passed to 'opt'")

    return output

def convm(x,p):

    '''Creates convolution matrix. Same as convm function in Matlab.'''

    if x.ndim != 1:
        AssertionError("Dimension Error: x must be 1d-array")

    x = x.astype(float)
    ppad = np.zeros((p-1,))
    xpad = np.append(x,ppad)
    ppad = np.append(xpad[0], ppad)
    convmat = toeplitz(xpad, ppad)

    return convmat

def covar(x,p):

    '''Creates covariance matrix. Same as covar in Matlab.'''

    if x.ndim != 1:
        AssertionError("Dimension Error: x must be 1d-array")

    x = x.astype(float)
    mu = [sum(x)/len(x)]*len(x)
    mu = np.asarray(mu)
    x = np.reshape(x-mu, (-1,1))
    con = convm(x,p)
    C = np.matmul(con.T,con)#/(m-1) # or no division by m-1?
    
    return C

def covar2(x,p):

    '''Creates covariance matrix. Same as covar in Matlab.'''

    if x.ndim != 1:
        AssertionError("Dimension Error: x must be 1d-array")

    x = x.astype(float)
    m = np.shape(x)[0]
    mu = np.ones((m,1))*(sum(x)/m)
    con = convm(x-mu,p)
    C = np.matmul(con.T,con)/(m-1) # or no division by m-1?
    
    return C

def blkconv(x):

    '''

    Creates convolution matrix of block elements from multivariate
    data.

    x: multi-dimensional data

    '''

    m, n = x.shape
    mrows = 2*m-1

    T = np.zeros((mrows, 1, m, n), dtype = x.dtype)
    xtup = tuple(x)
    for i, arr in enumerate(xtup):
        for j in range(m):            
            T[i + j, :, j, :] = arr
            
    T.shape = (mrows*1, m*n)
    
    return T

def BlkCovMat(x):

    '''
    
    Creates multivariate autocorrelation matrix, in which
    each element--R(0), R(1),...,R(L)--is a covariance matrix.
    Thus, the entire output is a block matrix.

    x: M x L matrix, where M represents the number of channels and 
       L represents the size of the lag.

    '''

    if type(x).__name__ != 'ndarray':
        AssertionError("Type Error: x must be an ndarray")

    if x.ndim > 2:
        AssertionError("Dimension Error: must be 2d-array")

    # Block mean Vector
    x = x.T
    mu = np.zeros(x.shape)
    nchans = len(x[0,:])
    for i in range(nchans):
        mu[:,i] = sum(x[:,i])/len(x[:,i])

    # Block Covariance Matrix
    diff = x - mu
    con = blkconv(diff)
    C = np.matmul(con.T, con)

    # Block Autocorrelation Matrix
    sz = C.shape
    mu = np.repeat(np.array([[mu[0,0],mu[0,1]]]), sz[1], axis=0)
    R = C + np.matmul(mu,mu.T)

    return (C, R)
