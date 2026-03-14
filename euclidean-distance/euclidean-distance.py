import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x = np.array(x)
    y = np.array(y)

    dist = np.sqrt(np.sum((x-y)**2))

    return float(dist)
   # L2 =0
    #for i in range(len(x)):
     #   L2 += (x[i]-y[i])**2
      #  return float(np.sqrt(L2))
    # Write code here
#    pass