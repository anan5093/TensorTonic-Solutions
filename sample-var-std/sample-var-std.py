import numpy as np

def sample_var_std(x):
    x = np.array(x)

    n= len(x)
    mean_x = np.mean(x)

    var = float(np.sum((x - mean_x) ** 2)/(n-1))
    std = float(np.sqrt(var))

    return(var, std)

    
    

   # pass