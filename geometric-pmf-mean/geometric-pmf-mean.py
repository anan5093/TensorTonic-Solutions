import numpy as np

def geometric_pmf_mean(k, p):

    k = np.array(k, dtype=float)
    p = float(p)

    pmf = p * ((1 - p) ** (k - 1))

    mean = 1 / p

    return (pmf.astype(float), float(mean))