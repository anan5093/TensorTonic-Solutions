import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    
    # positions column vector (T,1)
    positions = np.arange(seq_len)[:, None]
    
    # number of sin columns = ceil(d_model / 2)
    num_freq = (d_model + 1) // 2
    
    # frequency divisors (1, ceil(d/2))
    div_term = base ** (2 * np.arange(num_freq) / d_model)
    
    # angles via broadcasting (T, ceil(d/2))
    angles = positions / div_term
    
    # output matrix
    pe = np.zeros((seq_len, d_model), dtype=float)
    
    # even columns → sin
    pe[:, 0::2] = np.sin(angles)
    
    # odd columns → cos (slice to match number of cos cols)
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])
    
    return pe