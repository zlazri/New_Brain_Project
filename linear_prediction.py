import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve
from autocorrelation import AutoCorrMat
from autocorrelation import AutoCorr
from autocorrelation import CovMATLAB
from autocorrelation import covar
import pdb

def LinPredictor2(path, iseq, start, N, K, alp, skip):
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

    for i in range(start + K, start + N):

        # Autocorrelation Matrix   <---- Method 3
#        u = np.array(M[iseq, start:i+1])
#        C = covar(u, K)
#        mu = [sum(u)/len(u)]*K
#        mu = np.asarray(mu)
#        mu = np.reshape(mu, (-1,1))
#        Rextra = C + mu*mu.T
#        sz = len(Rextra[:,0])
#        R = Rextra[0:sz-1,0:sz-1]

        # Autocorrelation Matrix   <---- Method 3
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
            predicted = np.around(np.dot(np.flipud(w), M[iseq, i-K+alp:i+alp]))
            actual = M[iseq, i+alp]
            #print((predicted, actual))

            # Compute number of correct predictions
            if predicted == actual:
                correct += 1

            #Error
            #xi = R[0, 0] - np.dot(w, r)    #C[1:K, 0])
            #print("Error: " + str(xi))

        n += 1
 
    # Percentage of correct predictions
#    print("Correct: " + str(correct))
#    print("Total: " + str(total))
#    print("Percentage of correct predictions: "+ str(correct/total))

    p = correct/total

    return p


def LP(K,alp):
    # Longest seq: seq = 212, length = 1305, start_idx = 5673, end_idx = 6978

    p = LinPredictor2("M_positions_new.npy", 212, 5673, 1250, K, alp, 100)
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

    return max_len, max_indices

def maxseq(data):
    '''Finds the length of the longest continous data segment for each sequence in the data.  Among these, the longest segment is determined'''

    sz = len(data[:,0])-1
    max_segs = dict()
    
    for i in range(sz):
        max_segs[contsegs(data[i,:])[0]] = i

    max_len = max(max_segs.keys())
    max_seq = max_segs[max_len]

    return max_seq, contsegs(data[max_seq, :])

