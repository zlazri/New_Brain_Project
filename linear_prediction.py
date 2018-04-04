import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve
from autocorrelation import AutoCorrMat
from autocorrelation import AutoCorr
from autocorrelation import CovMATLAB
from autocorrelation import covar
from autocorrelation import convm
#from autocorrelation import BlkCovMat
from matplotlib import pylab as plt

def LinPredictor(path, iseq, start, N, K, alp, skip, return_vals=False):
    ''' Path: Path to the sequence data
        iseq: The index of the blob sequence
        start: The first frame of the sequence at which we will compute a blob location prediction
        N: Specifies the number of frames over which we wish to preform blob location estimations
        K: The number of previous frames taken into account to perform blob location estimate
        alp: Specifies how many frames into the future we wish to predict
        skip: Number of starting frames to skip over
    '''

    assert(alp > 0)

    # Initialize values
    M = np.load(path)
    n = 1
    correct = 0
    incorrect = 0
    total = N-skip
    Rextra = np.zeros((K+alp,K+alp))
    if return_vals == True:
        pred_arr = np.array([])
        act_arr = np.array([])

    for i in range(start + K, start + N):

        # Autocorrelation Matrix
        u = np.array(M[iseq, start:i+alp])
        C = covar(u, K+alp)
        mu = [sum(u)/len(u)]*(K+alp)
        mu = np.asarray(mu)
        mu = np.reshape(mu, (-1,1))
        Rextra = C + mu*mu.T
        sz = len(Rextra[:,0])
        R = Rextra[0:sz-alp,0:sz-alp]
        
        # Autocorrelation Vector
        r = Rextra[alp:sz,0]
        
        if i >= skip + start:
    
            # Set up system
            w = solve(R, r)

            # Filter
            predicted = np.around(np.dot(np.flipud(w), M[iseq, i-K:i]))
            actual = M[iseq, i+alp]
            #print((predicted, actual))

            # Compute number of correct predictions
            if predicted == actual:
                correct += 1

            # Output Arrays
            if return_vals == True:
                pred_arr = np.append(pred_arr, predicted)
                act_arr = np.append(act_arr, actual)

            #Error
            #xi = R[0, 0] - np.dot(w, r)    #C[1:K, 0])
            #print("Error: " + str(xi))

        n += 1
 
    # Percentage of correct predictions
#    print("Correct: " + str(correct))
#    print("Total: " + str(total))
#    print("Percentage of correct predictions: "+ str(correct/total))

    p = correct/total

    if return_vals == True:
        return p, pred_arr, act_arr
    else:
        return p

def dist_categorization(data_path1, data_path2, alp, K):
    '''Compares the correct predictions and incorrect predictions by how far of the predicted value is from the actual'''

    (p1, predict1, actual1) = LinPredictor(data_path1, 212, 5673, 1250, K, alp, 100, True)
    (p2, predict2, actual2) = LinPredictor(data_path2, 212, 5673, 1250, K, alp, 100, True)
    distance = abs(actual1-predict1) + abs(actual2-predict2)
    
    dist0 = 0
    dist1 = 0
    distelse = 0
    
    for i in range(len(distance)):
        if distance[i] == 0:
            dist0 += 1
        elif distance[i] == 1:
            dist1 += 1
        else:
            distelse += 1

    return (dist0/len(distance), dist1/len(distance), distelse/len(distance))

def hist_dist(data_path1, data_path2, alp, K):
    '''Returns multiple subplots of multi-histogram graphs depicting distributions of translation of predictions from actual result'''
    X = np.arange(K)
    fig, ax = plt.subplots(3,2, figsize=(10,10))
    for i in range(1,alp+1):
        print("Creating Plot " + str(i))  
        correct_arr = np.array([])
        dist1_arr = np.array([])
        distelse_arr = np.array([])
        for j in range(1, K+1):
            (dist0, dist1, distelse) = dist_categorization(data_path1, data_path2, i, j)
            correct_arr = np.append(correct_arr, dist0)
            dist1_arr = np.append(dist1_arr, dist1)
            distelse_arr = np.append(distelse_arr, distelse)
                 
        if np.mod(i, 2) > 0:
            m = int(np.ceil(i/2))
            n = 1
        else:
            m = int(i/2)
            n = 2
        
        line1 = ax[m-1, n-1].bar(X + 0.125, correct_arr, color = 'b', width = 0.25, label = "Correct")
        line2 = ax[m-1, n-1].bar(X + 0.375, dist1_arr, color = 'g', width = 0.25, label = "Off by 1")
        line3 = ax[m-1, n-1].bar(X + 0.625, distelse_arr, color = 'r', width = 0.25, label = "Off by distance >1")
        ax[m-1, n-1].set_xticks(X+0.5, [str(i) for i in range(1, K+1)])
        ax[m-1, n-1].set_ylim((0,1))
        ax[m-1, n-1].set_title('Predicting ' + str(i) + ' Frames into Future')

    plt.setp(ax[0,0].get_xticklabels(), visible=False)
    plt.setp(ax[0,1].get_xticklabels(), visible=False)
    plt.setp(ax[1,0].get_xticklabels(), visible=False)

    plt.setp(ax[0,1].get_yticklabels(), visible=False)
    plt.setp(ax[1,1].get_yticklabels(), visible=False)
     
    fig.delaxes(ax[2, 1])
    plt.figlegend((line1, line2, line3), ("Correct", "Off by 1", "Off by >1"), bbox_to_anchor=(.75, 0.3))
    fig.text(0.5, 0.03, 'Number of Previous Frames Taken into Account', ha='center', fontsize=12)
    fig.text(0.05, 0.5, 'Percent of Total Predictions', va='center', rotation='vertical', fontsize=12)
    plt.show()



    
def LPboxplot(data, K, alp, threshold):
    length = len(data[:,0])
    plt_dta = np.array([])
    for i in range(length-1):
        (maxlen, idxs) = contsegs(data[i, :])
        if maxlen >= threshold:
            plt_dta = np.append(plt_dta, LinPredictor("M_positions_new.npy", i, idxs[0], maxlen, K, alp, 20))

    # Create Boxplot

    plt.boxplot(plt_dta,vert=False)
    plt.title("Percentage of Correctly Predicted X Coordinates for All Sequences")
    plt.xlabel("Percent Correctly Predicted")    
    plt.show()
    

        
    
def LP(K,alp):
    # Longest seq: seq = 212, length = 1305, start_idx = 5673, end_idx = 6978

    p = LinPredictor("M_positions_new.npy", 212, 5673, 1250, K, alp, 100)
    return p

def continuous(seq):
    '''Takes a sequence and determines length of continuity by finding the first postion at which it reaches a NaN value'''
    
    for i in range(len(seq)):
        if np.isnan(seq[i]):
            output = i
            break
        else:
            output = i

    return output

def contsegs(seq):
    '''Finds the longest continuous segment in a sequence and returns the start and stop index of the sequence and its length'''

    rev_lens = dict()
    seq_idxs = dict()
    switch = False
    k = 1
    for i in range(len(seq)):
        if np.isnan(seq[i]):
            if switch == True:
                length = end_seq - start_seq
                rev_lens[length] = k
                seq_idxs[k] = (start_seq, end_seq)
                k += 1
                switch = False
            else:
                continue
        else:
            if switch == False:
                switch = True
                start_seq = i
                end_seq = i
            else:
                end_seq = i + 1

    max_len = max(rev_lens.keys())
    max_idx = rev_lens[max_len]
    max_indices = seq_idxs[max_idx]

    return (max_len, max_indices)

def maxseq(data):
    '''Finds the length of the longest continous data segment for each sequence in the data.  Among these, the longest segment is determined'''

    sz = len(data[:,0])-1
    max_segs = dict()
    
    for i in range(sz):
        max_segs[contsegs(data[i,:])[0]] = i

    max_len = max(max_segs.keys())
    max_seq = max_segs[max_len]

    return max_seq, contsegs(data[max_seq, :])

def MV_LP(x):

    ''' 
        Description: Uses Wiener filter for multivariate linear prediction
        
        x: k-channel signal of t-time series data

        i.e.

        x = [x_10 x_11 ... x_1L]
            [x_20 x_21 ... x_2L]
            [ .    .   .    .  ]
            [ .    .    .   .  ]
            [ .    .     .  .  ]
            [x_k0 x_k1     x_kL]

        RL = [R(0)      R(1)    ...  R(L-1)]
             [R(1).T    R(0)    ...  R(L-2)]
             [  .        .      .      .   ] 
             [  .        .       .     .   ]
             [  .        .        .    .   ]
             [R(L-1).T R(L-2).T ...  R(0)  ]

        Rf = [R(1) R(2) ... R(L)].T

    '''
    (mdim, ndim) = x.shape
    (C, R) = BlkCovMat(x)
    blk = mdim
    nR = len(R[:,0])
    Rf = R[blk:nR, 0:blk]
    L  = int(nR/blk - 1)
    
    # Initialization
    Ef_old = R[0:blk, 0:blk]
    Eb_old = R[0:blk, 0:blk]
    I = np.identity(blk)
    Z = np.zeros((blk,blk))
    Kb = R[blk:blk*2, 0:blk]
    A_old = Z + np.matmul(np.matmul(I, inv(Eb_old)), Kb.T)
    B_old = Z + np.matmul(np.matmul(I, inv(Ef_old)), Kb)
    Ef_old = Ef_old - np.matmul(np.matmul(Kb, inv(Eb_old)), Kb.T)
    Eb_old = Eb_old - np.matmul(np.matmul(Kb.T, inv(Ef_old)), Kb)
    
    for i in range(2, L+1):
        
        # Recursion
        R_l = R[blk*i:blk*(i+1), 0:blk]
        Rf_l = R[blk:blk*i, 0:blk]
        Kb = R_l - np.matmul(Rf_l.T, B_old)
            
        A = np.block([[A_old], [Z]]) - np.matmul(np.block([[B_old], [-I]]), np.matmul(inv(Eb_old), Kb.T))
        B = np.block([[Z], [B_old]]) - np.matmul(np.block([[-I], [A_old]]), np.matmul(inv(Ef_old), Kb))
        Ef = Ef_old - np.matmul(Kb, np.matmul(inv(Eb_old), Kb.T))
        Eb = Eb_old - np.matmul(Kb.T, np.matmul(inv(Ef_old), Kb))

        # Update old matrices
        A_old = A
        B_old = B
        Eb_old = Eb
        Ef_old = Ef
        
    xvec = np.reshape(x[:,::-1].T, (mdim*(ndim),1))
    pred = np.around(np.matmul(A.T, xvec[2:]))

    return pred

# TODO: Check the initialization updates that I made above!!!
# I think I'm almost there


def MV_accuracy(data, start, N, filtsz = None):

    ''' data: the full sequence
        start: where predictions start (cannot be first index. >100 recommeneded)
        N: number of predictions
    '''

    total = 0
    correct = 0
    calculated = 0
    for i in range(start,start+N):
        if filtsz == None:
            sample = data[:, 0:i]
        else:
            sample = data[:, (i-filtsz):i]
        pred = MV_LP(sample)
        prediction = (pred[0], pred[1])
        actual = (data[0, i+1], data[1, i+1])
        if actual == prediction:
            correct += 1
        total += 1
        calculated += 1
        print(str(actual)+str(prediction))
        print(str(calculated) + " predictions calculated")
    percent = correct/total
    return percent

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
