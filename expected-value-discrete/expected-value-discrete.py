import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.array(x, dtype=float)
    p = np.array(p, dtype=float)

    # Check same length
    if x.shape != p.shape:
        raise ValueError("x and p must have same length")

    # Check probabilities valid
    if np.any(p < 0) or not np.isclose(np.sum(p), 1):
        raise ValueError("Invalid probabilities")

    Et = np.sum(x * p)
    return float(Et)