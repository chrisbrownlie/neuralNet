import numpy as np
from activations import sigmoid

## Implement basic neural network
class basicNet:
    """
    Simple neural network for classification
    """

    def __init__(self, x, y, hidden_neurons, activation, alpha):
        self.x = x
        self.y = y
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.alpha = alpha
        self.layers = [x.shape[1], hidden_neurons, y.shape[0]]


    def initialise(self):
        """
        Initialise random network weights and biases
        """
        params = [0]
        print("Initialising network parameters...")
        for i, v in enumerate(self.layers):
            if i == 0:
                break
            print("Initialising random weights for layer " + str(i))
            weights = np.random.random((v, self.layers[i-1]))
            bias = np.zeros((v, 1))

            params.append([weights, bias])

        self.params = params
        print(params)

    def forward_propagate(self):
        """
        Perform one forward propagation
        """
        cache = [[self.x, np.transpose(self.x)]]

        for i, v in enumerate(self.layers):
            if i == 0:
                break
            print("Propagating from layer " + str(i-1) + " to layer " + str(i))

            b_matrix = self.params


eg_x = np.random.random((100,4))
eg_y = np.random.random((100,1))
test = basicNet(eg_x, eg_y, 6, "sigmoid", 0.2)

test.initialise()

