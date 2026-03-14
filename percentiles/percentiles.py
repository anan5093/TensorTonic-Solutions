import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x = np.array(x, dtype=float)
    q = np.array(q, dtype=float)

    if x.size == 0:
        raise ValueError("x cannot be empty")

    if np.any((q < 0) | (q > 100)):
        raise ValueError("Percentiles must be in [0, 100]")

    x = np.sort(x)

    result = np.percentile(x, q, method='linear')

    return np.array(result)