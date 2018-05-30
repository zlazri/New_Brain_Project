import numpy as np
from scipy.linalg import toeplitz
from scipy.linalg import inv
from matplotlib import pyplot as plt

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

    pred_arr = []
    actual_arr = []
    
    mseqs = np.load("M_positions_new.npy")
    nseqs = np.load("N_positions_new.npy")
    M = mseqs[7, 0:400]
    N = nseqs[7, 0:400]

#    Test Data
#    M = np.linspace(1, 100, 400)
#    N = np.linspace(0, 2*np.pi, 400)
#    N = np.sin(N)

    for i in range(100):
        Msamp = M[i:200+i]
        Nsamp = N[i:200+i]

        # Predicted
        pred = MV_Wiener(Msamp, Nsamp)

        # Actual
        actual = np.array([[M[200+i]],[N[200+i]]])

        # Percent Correct
        if np.around(pred)[0] == actual[0] and np.around(pred)[1] == actual[1]:
            correct += 1
        total += 1
    print(str(correct/total))
    
#        # Create arrays
#        pred_arr.append((pred[0][0], pred[1][0]))
#        actual_arr.append((M[200+i], N[200+i]))
#
#    pred_arr = np.asarray(pred_arr)
#    actual_arr = np.asarray(actual_arr)

#    print(pred_arr)
#    print(actual_arr)

#    # Plots
#    plt.plot(pred_arr[:,0], pred_arr[:,1])
#    plt.plot(actual_arr[:,0], actual_arr[:,1])
#    plt.show()
