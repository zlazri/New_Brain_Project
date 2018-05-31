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
    sz = int((x.size))
    x = x-np.mean(x)
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

def MV_Wiener(M, N, lags):
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
    R = Covar(x, lags)   # <----- play around with second paramameter
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

def multipredictor(M, N, sz, n, lags):
    correct = 0
    total = 0

    pred_arr = []
    actual_arr = []
    
    for i in range(n):
        Msamp = M[i:sz+i]
        Nsamp = N[i:sz+i]

        # Predicted
        pred = MV_Wiener(Msamp, Nsamp, lags)
        #pred = np.around(pred)   #<---- Use if you want rounding

        # Actual
        actual = np.array([[M[sz+i]],[N[sz+i]]])

        # Create arrays
        pred_arr.append((pred[0][0], pred[1][0]))
        actual_arr.append((M[sz+i], N[sz+i]))

        # Percent Correct
        if pred[0] == actual[0] and pred[1] == actual[1]:
            correct += 1
        total += 1

    pred_arr = np.asarray(pred_arr)
    actual_arr = np.asarray(actual_arr)
        
    return actual_arr, pred_arr, correct/total

if __name__ == "__main__":

    mseqs = np.load("M_positions_new.npy")
    nseqs = np.load("N_positions_new.npy")
    M = mseqs[7, 0:401]
    N = nseqs[7, 0:401]

    # Test Data
#    M = np.linspace(0, 400, 401)
#    N = M

    (act, pred, p) = multipredictor(M, N, 200, 100, 5)
    print("Percent Correct: " + str(p))

    # Subplots

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True)
    ax1.plot(act[:, 0], act[:, 1], color = 'blue', label="True Trajectory")
    ax2.plot(pred[:, 0], pred[:, 1], color = 'orange', label="Predicted Trajectory")
    ax3.plot(act[:, 0], act[:, 1], color = 'blue', label="True Trajectory")
    ax3.plot(pred[:, 0], pred[:, 1], color = 'orange', label="Predicted Trajectory")
    ax1.set_ylabel("Y")
    ax2.set_xlabel("X")
    ax1.legend(loc = "upper right")
    ax2.legend(loc = "upper right")
    ax3.legend(loc = "upper right")
    fig.suptitle("Actual vs Predicted Trajectory of Line")
    plt.show()
