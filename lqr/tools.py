
import math
import numpy as np

def smat(x):
    # estimates what `n` s.t. n(n+1)/2 = len(x)
    # hacky fix
    mid = (2*len(x))**0.5
    n = math.floor(mid)
    
    if int(n*(n+1)/2) != len(x):
        print("len(x) must be sum of 1,2,...,n. Given length, expected {} but got {}".format(
            int(n*(n+1)/2), len(x)
        ))
        assert False
        
    ct = 0
    X = np.zeros((n,n))
    for i in range(n):
        X[i,i] = x[ct]
        ct += 1
        for j in range(i+1,n):
            X[i,j] = X[j,i] = x[ct] / (2**0.5)
            ct += 1
            
    return X

def svec(X):
    n = X.shape[0]
    nn = int(n*(n+1)/2)
    x = np.zeros(nn)
    
    (i,j) = (0,0)
    for ct in range(nn):
        if i==j:
            x[ct] = X[i,j]
        else:
            x[ct] = X[i,j] * (2**0.5)
        ct += 1
        j += 1
        if j >= n:
            i += 1
            j = i
    return x