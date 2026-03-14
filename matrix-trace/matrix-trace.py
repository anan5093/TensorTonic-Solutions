import numpy as np

def matrix_trace(A):
    A = np.array(A)
    
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square")
    
    trace = 0
    for i in range(n):
        trace += A[i, i]
        
    return float(trace)