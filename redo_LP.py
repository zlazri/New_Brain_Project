import numpy as np
from scipy.linalg import toeplitz
from scipy.linalg import inv

def reverse(x):
    x = x[::-1]
    return x
    
def mean_zero(x):
    mu = np.mean(x)
    x_mu0 = x - mu
    return x_mu0

def restruct(x):
    sz = x.size
    x = np.reshape(x, (sz,1), order = 'F')
    return x

def Covar(x, p):
    sz = int((x.size)/2)
    con = Convm(x,p)
    R = np.matmul(con.T, con)/(sz-1)
    return R
     
def Convm(x,p):

    '''Creates convolution matrix. Same as convm function in Matlab.'''

    if x.ndim != 1:
        AssertionError("Dimension Error: x must be 1d-array")

    x = x.astype(float)
    p = 2*p
    ppad = np.zeros((p-1,))
    xpad = np.append(x,ppad)
    ppad = np.append(xpad[0], ppad)
    convmat = toeplitz(xpad, ppad)

    return convmat

def MV_Wiener(M, N):
    '''Inputs: M and N are the data from two channels of multivariate data.'''

    # Reverse seqs
    Mrev = reverse(M)
    Nrev = reverse(N)

    # Convert seqs to mean zero
    Mmu0 = mean_zero(Mrev)
    Nmu0 = mean_zero(Nrev)

    # Create multichannel vector
    x = np.array([Mmu0, Nmu0])
    x = restruct(x)

    # Create System (find R matrix and r vector)
    R = Covar(x, 5)   # <----- play around with second paramameter
    L = R.shape[0]
    r = R[0:2, 2:].T
    RL = R[0:L-2, 0:L-2]  # <---- since we are using 2 channels, subtract 2 instead of 1!!!!!

    # Solve System
    Rinv = inv(RL)
    A = np.matmul(Rinv, r)
    
    # Predict
    xL = np.array([x[0:L-2, 0]]).T
    muM = np.mean(M)
    muN = np.mean(N)
    mu = np.array([[muM,muN]]).T
    pred = np.matmul(A.T, xL)
    pred = pred + mu
    
    return pred

if __name__ == "__main__":

    correct = 0
    total = 0
    
    mseqs = np.load("M_positions_new.npy")
    nseqs = np.load("N_positions_new.npy")
    M = mseqs[7, 0:400]
    N = nseqs[7, 0:400]
    for i in range(100):
        Msamp = M[i:200+i]
        Nsamp = N[i:200+i]

        # Predicted
        pred = MV_Wiener(Msamp, Nsamp)

        # Actual
        actual = np.array([[M[200+i]],[N[200+i]]])

        # Compare results
        if np.around(pred)[0] == actual[0] and np.around(pred)[1] == actual[1]:
            correct += 1
        total += 1
    print(str(correct/total))
    






















    
#if __name__ == "__main__":
#
#    mseqs = np.load("M_positions_new.npy")
#    nseqs = np.load("N_positions_new.npy")
#    M = mseqs[7, 0:400]
#    N = nseqs[7, 0:400]
#    Msamp = M[0:200]
#    Nsamp = N[0:200]
#
#    # Reverse seqs
#    Msamp = reverse(Msamp)
#    Nsamp = reverse(Nsamp)
#
#    # Convert seqs to mean zero
#    Mmu0 = mean_zero(Msamp)
#    Nmu0 = mean_zero(Nsamp)
#
#    # Create multichannel vector
#    x = np.array([Mmu0, Nmu0])
#    x = restruct(x)
#
#    # Create System (find R matrix and r vector)
#    R = Covar(x, 5)   # <----- play around with second paramameter
#    L = R.shape[0]
#    r = R[0:2, 2:].T
#    RL = R[0:L-2, 0:L-2]  # <---- since we are using 2 channels, subtract 2 instead of 1!!!!!
#
#    # Solve System
#    Rinv = inv(RL)
#    A = np.matmul(Rinv, r)
#    
#    # Predict
#    xL = np.array([x[0:L-2, 0]]).T
#    muM = np.mean(Msamp)
#    muN = np.mean(Nsamp)
#    mu = np.array([[muM,muN]]).T
#    pred = np.matmul(A.T, xL)
#    pred = pred + mu
#    actual = np.array([[M[200], N[200]]]).T
#    print(pred)
#    print(actual)
