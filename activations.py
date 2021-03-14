import numpy as np
def sigmoid(a, d = False):
    """
    Sigmoid activation function (d = False) and its derivative (d = True)
    """
    if d:
        return a*(1-a)
    else:
        return 1/(1+np.exp(-a))

def relu(a, d = False):
    """
    Rectified linear unit activation function
    """
    return np.maximum(0, a)