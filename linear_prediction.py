import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve
from autocorrelation import AutoCorrMat
from autocorrelation import AutoCorr
from autocorrelation import CovMATLAB
from autocorrelation import covar
from autocorrelation import convm
from matplotlib import pylab as plt
from autocorrelation import covar2
from blkT import blkT

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
        mu = np.ones((K+alp, 1))*(sum(u)/len(u))
        Rextra = C + (mu*mu.T)#/(len(u)-1)
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
    p = correct/total

    if return_vals == True:
        return p, pred_arr, act_arr
    else:
        return p

def dist_categorization(data_path1, data_path2, alp, K):
    '''Compares the correct predictions and incorrect predictions by how far of the predicted value is from the actual'''

    (p1, predict1, actual1) = LinPredictor(data_path1, 212, 5673, 1000, K, alp, 100, True)
    (p2, predict2, actual2) = LinPredictor(data_path2, 212, 5673, 1000, K, alp, 100, True)
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

def MV_accuracy(data, start, N, K, sampsz = None, show = True):

    ''' data: the full sequence
        start: where predictions start (cannot be first index. >100 recommeneded)
        N: number of predictions
    '''

    total = 0
    correct = 0
    calculated = 0
    pred_arr = []
    act_arr = []
    for i in range(start,start+N):
        if sampsz == None:
            sample = data[:, 0:i]
        else:
            sample = data[:, (i-sampsz):i]

        # Predicted Value    
        pred = MV_LP(sample[:,::-1], K)
        prediction = (pred[0][0], pred[0][1])
        pred_arr.append(prediction)

        # Actual Value
        actual = (data[0, i+1], data[1, i+1])
        act_arr.append(actual)

        # Percent Calculations
        if actual == prediction:
            correct += 1
        total += 1
        calculated += 1
        if show == True:
            print(str(calculated) + " predictions calculated")
            
    percent = correct/total
    pred_arr = np.asarray(pred_arr)
    act_arr = np.asarray(act_arr)
    
    return (act_arr, pred_arr), percent

def MV_LP(x, K):

    ''' 
        Description: Uses Wiener filter for multivariate linear prediction
        
        x: k-channel signal of t-time series data

        K: number of previous frames taken into account

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

        Note: Block element transpose only supported for channels of size 2. Need to create a new function for larger channel arrays.

    '''
    (mdim, ndim) = x.shape
#    (C, R) = CorrCovMats(x, K)
    C  = CorrCovMats(x,K)
    blk = mdim

    nR = len(C[0,:])
    Rf = C[0:blk, blk:nR]
    RL = C[0:nR-blk, 0:nR-blk]
    Rf = Rf.T
    Rinv = inv(RL)
    A = np.matmul(Rinv, Rf)
    x = x[:,0:K-1]
    xvec = np.reshape(x.T, ((mdim*K)-2, 1))
    #pred = np.around(np.matmul(A.T, xvec))
    pred = np.matmul(A.T, xvec) # <------ no rounding
    pred = pred.T
    
#    nR = len(R[0,:])
#    Rf = R[0:blk, blk:nR]
#    RL = R[0:nR-blk, 0:nR-blk]
#    Rf = Rf.T
#    Rinv = inv(RL)
#    A = np.matmul(Rinv, Rf)
#    x = x[:,0:K-1]
#    xvec = np.reshape(x.T, ((mdim*K)-2, 1))
#    #pred = np.around(np.matmul(A.T, xvec))
#    pred = np.matmul(A.T, xvec) # <------ no rounding
    
    return np.around(pred)

def CorrCovMats(x, K):

    '''
    
    Creates multivariate autocorrelation matrix, in which
    each block element--R(0), R(1),...,R(L)--is a autocorrelation matrix.

    x: M x L matrix, where M represents the number of channels and 
       L represents the size of the lag.

    '''

    if type(x).__name__ != 'ndarray':
        AssertionError("Type Error: x must be an ndarray")

    if x.ndim > 2:
        AssertionError("Dimension Error: must be 2d-array")

    xmu = sum(x[0,:])/len(x[0,:])
    ymu = sum(x[1,:])/len(x[1,:])
    mu = np.array([[xmu], [ymu]])
    mu = np.repeat(mu, x.shape[1], axis=1)
    mu0 = x - mu
    mu0 = mu0.T
    (mdims, ndims) = mu0.shape
    mu0 = np.reshape(mu0, ((mdims*ndims, 1)), 0)
    length = mu0.size
    C = covar2(mu0, K*2)/(length-1)
    #x = x.T
    #(mdims,ndims) = x.shape
    #x = np.reshape(x, ((mdims*ndims, 1)), 0)
    #xlen = x.size
    #m = np.sum(x)/xlen
    #mu = np.ones((K*2,1))*m
    #C = covar2(x-m, K*2)/(xlen-1)
    #R = C + (mu*mu.T)/(xlen-1)
    
    #return (C, R)

    return C

def trajectory(data, start, N, K, sampsz):

    "Description: Plots the trajectories of two sequences on one graph"
    

    ((data1, data2), p) = MV_accuracy(data, start, N, K, sampsz)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    ax1.plot(data1[:,0], data1[:,1], c = 'b')
    ax2.plot(data2[:,0], data2[:,1], c = 'r')
    ax3.plot(data1[:,0], data1[:,1], c = 'b')
    ax3.plot(data2[:,0], data2[:,1], c = 'r')

    ax1.title.set_text("Actual Trajectory")
    ax2.title.set_text("Predicted Trajectory")
    ax3.title.set_text("Overlapping Trajectories")

    ax1.set_ylabel("Y position")
    ax2.set_xlabel("X position")

    fig.tight_layout()
    
    plt.show()

def pred_correct(data, start, N, k, sampsz):

    ''' Graphs predictions against actual results'''
    ((data1, data2), p1) = MV_accuracy(data, start, N, k[0], sampsz)
    ((data3, data4), p2) = MV_accuracy(data, start, N, k[1], sampsz)
    ((data5, data6), p3) = MV_accuracy(data, start, N, k[2], sampsz)
    plt.legend(loc=1, fontsize="small")

    
    frames = np.linspace(start, start + N, N)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)

    # X coordinates plot
    ax1.plot(frames, data1[:,0], label="Actual")
    ax1.plot(frames, data2[:,0], label="Predicted")
    ax1.set_xlabel("Time (Frame #)")
    ax1.set_title("Predicted vs Actual X Positions, L = " + str(k[0]))
    ax1.legend(loc=1, fontsize="small")

    # Y coordinates plt
    ax2.plot(frames, data1[:,1])
    ax2.plot(frames, data2[:,1])
    ax2.set_xlabel("Time (Frame #)")
    ax2.set_title("Predicted vs Actual Y Positions, L = " + str(k[0]))

    # X coordinates plot
    ax3.plot(frames, data3[:,0])
    ax3.plot(frames, data4[:,0])
    ax3.set_xlabel("Time (Frame #)")
    ax3.set_title("Predicted vs Actual X Positions, L = " + str(k[1]))

    # Y coordinates plt
    ax4.plot(frames, data3[:,1])
    ax4.plot(frames, data4[:,1])
    ax4.set_xlabel("Time (Frame #)")
    ax4.set_title("Predicted vs Actual Y Positions, L = " + str(k[1]))

    # X coordinates plot
    ax5.plot(frames, data5[:,0])
    ax5.plot(frames, data6[:,0])
    ax5.set_xlabel("Time (Frame #)")
    ax5.set_title("Predicted vs Actual X Positions, L = " + str(k[2]))

    # Y coordinates plt
    ax6.plot(frames, data5[:,1])
    ax6.plot(frames, data6[:,1])
    ax6.set_xlabel("Time (Frame #)")
    ax6.set_title("Predicted vs Actual Y Positions, L = " + str(k[2]))

    fig.tight_layout()

    plt.show()

def percent_graphs(data, start, N, k, sample):

    # k is array of lags. k-elements > 1

    assert(min(k)>1), "ValueError: k must be greater than 1"
    
    coor_arr = np.array([])
    off1_arr = np.array([])
    off2_arr = np.array([])
    offbig_arr = np.array([])
    
    for K in k:
        print("Performing Calculations for lag: " + str(int(K)))
        (correct, off1, off2, offbig) =  MVpercents(data, start, N, int(K), sample)
        print(correct)
        print(off1)
        print(off2)
        print(offbig)
        coor_arr = np.append(coor_arr, [correct])
        off1_arr = np.append(off1_arr, [off1])
        off2_arr = np.append(off2_arr, [off2])
        offbig_arr = np.append(offbig_arr, [offbig])
        
    # Graphs
   # xticks = np.linspace(2, k, k-2+1)
    plt.plot(k, coor_arr, label="Correct")
    plt.plot(k, off1_arr, label = "Off 1")
    plt.plot(k, off2_arr, label = "Off 2")
    plt.plot(k, offbig_arr, label = "Off >2")
    plt.xticks(k)
    plt.legend(loc=1, fontsize="small")
    plt.xlabel("Number of Frame Lags Taken into Account")
    plt.ylabel("Prediction Accuracy (%)")
    plt.title("Accuracy of Multivariate Prediction at Different Time Lags")
    
    plt.show()    
    
def MVpercents(data, start, N, K, sampsz):
    
    ''' Bins the distances between a sample of actual and predicted values
    (correct, off by 1, off by 2, off by >2)

    data: multivariate data
    start: the index at which we start predicting
    N: number of predictions
    K: array of lags
    d: distance between actual and predicted point
    '''

    total = 0
    correct = 0
    off1 = 0
    off2 = 0
    offbig = 0
    calculated = 0
    pred_arr = []
    act_arr = []
    for i in range(start, start + N):
        sample = data[:, (i-sampsz):i]
        
        # Predicted Value    
        pred = MV_LP(sample[:, ::-1], K)
        prediction = np.array([pred[0][0], pred[0][1]])
        pred_arr.append(prediction)

        # Actual Value
        actual = np.array([data[0, i+1], data[1, i+1]])
        act_arr.append(actual)

        # Percent Calculations
        if sum(np.sqrt(np.square(actual-prediction))) == 0:
            correct += 1
        elif sum(np.sqrt(np.square(actual-prediction))) == 1:
            off1 += 1
        elif sum(np.sqrt(np.square(actual-prediction))) == 2:
            off2 += 1
        else:
            offbig += 1
        total += 1
        calculated += 1

    corr_per = correct/total
    off1_per = off1/total
    off2_per = off2/total
    offbig_per = offbig/total

    return(corr_per, off1_per, off2_per, offbig_per)


#-------------------------Scrap/Test Code----------------------------------

''' Function 1 '''

#def BlkCovMat(x):
#
#    '''
#    
#    Creates multivariate autocorrelation matrix, in which
#    each block element--R(0), R(1),...,R(L)--is a autocorrelation matrix.
#
#    x: M x L matrix, where M represents the number of channels and 
#       L represents the size of the lag.
#
#    '''
#
#    if type(x).__name__ != 'ndarray':
#        AssertionError("Type Error: x must be an ndarray")
#
#    if x.ndim > 2:
#        AssertionError("Dimension Error: must be 2d-array")
#
#    # Block mean Vector
#    x = x.T
#    mu = np.zeros(x.shape)
#    nchans = len(x[0,:])
#    for i in range(nchans):
#        mu[:,i] = sum(x[:,i])/len(x[:,i])
#
#    # Block Covariance Matrix
#    diff = x - mu
#    con = blkconv(diff)
#    C = np.matmul(con.T, con)/len(x[:,i])
#    
#    # Block Autocorrelation Matrix
#    conmu = blkconv(mu)
#    mu_sq = np.matmul(conmu.T, conmu)
#    sz = C.shape
#    mux = mu[0,0]
#    muy = mu[0,1]
#    mu_sq = np.array([[mux*mux, mux*muy], [muy*mux, muy*muy]])
#    mat1 = np.ones((int(sz[0]/2),int(sz[1]/2)))
#    mu_sq = np.kron(mat1,mu_sq)
#    print(mu_sq)
#    R = C + mu_sq
#    mux = np.array([mu[:,0]]).T
#    muy = np.array([mu[:,1]]).T
#    mu_sq = np.matmul(mux, muy.T)
#    R = C + mu_sq
#
#    return (C, R)

''' Fucntion 2 '''

#def blkconv(x):
#
#    '''
#
#    Creates convolution matrix of block elements from multivariate
#    data.
#
#    x: multi-dimensional data
#
#    '''
#
#    m, n = x.shape
#    mrows = 2*m-1
#
#    T = np.zeros((mrows, 1, m, n), dtype = x.dtype)
#    xtup = tuple(x)
#    for i, arr in enumerate(xtup):
#        for j in range(m):            
#            T[i + j, :, j, :] = arr
#            
#    T.shape = (mrows*1, m*n)
#    
#    return T

''' Function 3 '''

#def MV_LP_lev(x):
#
#    ''' 
#        Description: Uses Wiener filter for multivariate linear prediction
#        
#        x: k-channel signal of t-time series data
#
#        i.e.
#
#        x = [x_10 x_11 ... x_1L]
#            [x_20 x_21 ... x_2L]
#            [ .    .   .    .  ]
#            [ .    .    .   .  ]
#            [ .    .     .  .  ]
#            [x_k0 x_k1     x_kL]
#
#        RL = [R(0)      R(1)    ...  R(L-1)]
#             [R(1).T    R(0)    ...  R(L-2)]
#             [  .        .      .      .   ] 
#             [  .        .       .     .   ]
#             [  .        .        .    .   ]
#             [R(L-1).T R(L-2).T ...  R(0)  ]
#
#        Rf = [R(1) R(2) ... R(L)].T
#
#        Note: Block element transpose only supported for channels of size 2. Need# to create a new function for larger channel arrays.
#
#    '''
#    (mdim, ndim) = x.shape
#    (C, R) = BlkCovMat(x)
#    blk = mdim
#    nR = len(R[0,:])
#    Rf = R[0:blk, blk:nR]
#    Rf = blkT(Rf)
#    L  = int(nR/blk - 1)
#    
#    # Initialization
#    Ef_old = R[0:blk, 0:blk]
#    Eb_old = R[0:blk, 0:blk]
#    I = np.identity(blk)
#    Z = np.zeros((blk,blk))
#    Kb = R[blk:blk*2, 0:blk]
#    A_old = Z + np.matmul(np.matmul(I, inv(Eb_old)), Kb.T)
#    B_old = Z + np.matmul(np.matmul(I, inv(Ef_old)), Kb)
#    Ef_old = Ef_old - np.matmul(np.matmul(Kb, inv(Eb_old)), Kb.T)
#    Eb_old = Eb_old - np.matmul(np.matmul(Kb.T, inv(Ef_old)), Kb)
#    
#    for i in range(2, L+1):
#        
#        # Recursion
#        R_l = R[blk*i:blk*(i+1), 0:blk]
#        #Rf_l = R[0:blk, blk:blk*i].T
#        Rf_l = Rf[0:blk*(i-1), :]
#        Kb = R_l - np.matmul(Rf_l.T, B_old)
#            
#        A = np.block([[A_old], [Z]]) - np.matmul(np.block([[B_old], [-I]]), np.ma#tmul(inv(Eb_old), Kb.T))
#        B = np.block([[Z], [B_old]]) - np.matmul(np.block([[-I], [A_old]]), np.ma#tmul(inv(Ef_old), Kb))
#        Ef = Ef_old - np.matmul(Kb, np.matmul(inv(Eb_old), Kb.T))
#        Eb = Eb_old - np.matmul(Kb.T, np.matmul(inv(Ef_old), Kb))
#
#        # Update old matrices
#        A_old = A
#        B_old = B
#        Eb_old = Eb
#        Ef_old = Ef
#
#    x = x[:,0:ndim-1]
#    xvec = np.reshape(x[:,::-1].T, ((mdim*ndim)-2,1))
#    pred = np.around(np.matmul(A.T, xvec))
#
#    return pred.T
#
# TODO: Fix the R-block elements. They may al be transposed.

''' Function 4 '''

#def LPboxplot(data, K, alp, threshold):
#    length = len(data[:,0])
#    plt_dta = np.array([])
#    for i in range(length-1):
#        (maxlen, idxs) = contsegs(data[i, :])
#        if maxlen >= threshold:
#            plt_dta = np.append(plt_dta, LinPredictor("M_positions_new.npy", i, id#xs[0], maxlen, K, alp, 20))
#
#    # Create Boxplot
#
#    plt.boxplot(plt_dta,vert=False)
#    plt.title("Percentage of Correctly Predicted X Coordinates for All Sequences"#)
#    plt.xlabel("Percent Correctly Predicted")    
#    plt.show()
