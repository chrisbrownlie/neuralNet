import numpy as np
def sigmoid(a):
    """
    Sigmoid activation function
    """
    1/(1+np.exp(a))