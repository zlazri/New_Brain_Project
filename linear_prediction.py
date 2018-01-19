import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve

M = np.load('M_positions.npy')

N = 200
K = 2

n = 1
for i in range(K - 1, N):
    # Compute sample mean vector
    mu = np.zeros((K, 1))
    for j in range(K - 1, i + 1):
        u = np.array([M[1, j-K+1:j+1]]).T
        mu += u
    mu /= n

    # Compute sample covariance/autocorrelation matrix
    C = np.zeros((K, K))
    for j in range(K - 1, i + 1):
        u = np.array([M[1, j-K+1:j+1]]).T
        C += (u - mu)*(u - mu).T
        # C += u*u.T
    C /= n
    # print(C)
    C += mu*mu.T

    # print(u)
    # print(C)

    if i > 100:
        # Set up system
        R = C[0:K-1,0:K-1]
        r_x = C[1:K, 0]
        w = solve(R, r_x)
        # print(w)

        # Filter
        print((np.dot(np.flipud(w), M[1, i-K+2:i+1]), M[1, i]))

        xi = C[0, 0] - np.dot(w, C[1:K, 0])
        # print(xi)
    n += 1
 
 
# TODO: Now do the multi-step method (or add it as an option to the code above).
# See page 346 of textbook.
