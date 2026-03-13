import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    
    # convert to numpy arrays (safe if already numpy)
    a = np.array(a)
    b = np.array(b)
    
    # dot product
    dot = np.sum(a * b)
    
    # norms
    norm_a = np.sqrt(np.sum(a * a))
    norm_b = np.sqrt(np.sum(b * b))
    
    # handle zero vector case
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)
    