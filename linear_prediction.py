import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve

def LinPredictor(path, nseq, start, N, K, alp, skip):
    ''' Path: Path to the sequence data
        nseq: The index of the blob sequence
        start: The first frame of the sequence at which we will compute a blob location prediction
        N: Specifies the number of frames over which we wish to preform blob location estimations
        K: The number of previous frames taken into account to perform blob location estimate
        alp: Specifies how many frames into the future we wish to predict
        skip: Number of starting frames to skip over
    '''
    
    M = np.load(path)
    n = 1
    correct = 0
    incorrect = 0
    total = N-skip
    
    for i in range(start + K - 1, start + N):
        # Compute sample mean vector
        mu = np.zeros((K + alp, 1))
        for j in range(start + K - 1, i + 1):
            u = np.array([M[nseq, j-K+1:j+1+alp]]).T
            mu += u
        mu /= n

        # Compute sample covariance/autocorrelation matrix
        C = np.zeros((K + alp, K + alp))
        for j in range(start + K - 1, i + 1):
            u = np.array([M[nseq, j-K+1:j+1+alp]]).T
            C += (u - mu)*(u - mu).T
            # C += u*u.T
        C /= n
        # print(C)
        C += mu*mu.T

        # print(u)
        # print(C)

        if i >= skip:
            # Set up system
            R = C[0:K-1,0:K-1]
            r_x = C[1+alp:K+alp, 0]
            w = solve(R, r_x)
            # print(w)

            # Filter
            predicted = np.around(np.dot(np.flipud(w), M[nseq, i-K+2:i+1]))
            actual = M[nseq, i+alp]
            print((predicted, actual))

            # Compute number of correct predictions
            if predicted == actual:
                correct += 1
                        
            #xi = C[0, 0] - np.dot(w, r_x)    #C[1:K, 0])
            #print(xi)
        n += 1
 
 
        # Percentage of correct predictions
    print("Percentage of correct predictions: "+ str(correct/total))
