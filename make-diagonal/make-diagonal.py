import numpy as np

def make_diagonal(v):
    V = np.array(v)
    n = V.shape[0]
    
    D = np.zeros((n, n))
    for i in range(n):
        D[i][i] = V[i]
        
    return D