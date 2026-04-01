import numpy as np
from collections import Counter

def mean_median_mode(x):
    x = np.array(x)

    mean_ = float(np.mean(x))
    median_ = float(np.median(x))

    freq_ =Counter(x)
    max_count = max(freq_.values())
    mode_ = float(min([k for k, v in freq_.items() if v == max_count]))

    return(mean_, median_, mode_)
    #pass