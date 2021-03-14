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
        self.layers = [x.shape[1], hidden_neurons, y.shape[1]]


    def initialise(self):
        """
        Initialise random network weights and biases
        """
        params = [0]
        print("Initialising network parameters...")
        for i, v in enumerate(self.layers):
            if i == 0:
                continue
            print("Initialising random weights for layer " + str(i))
            weights = np.random.random((v, self.layers[i-1]))
            print("Weights have shape " + str(weights.shape))
            bias = np.zeros((1, v))

            params.append([weights, bias])

        self.params = params

    def forward_propagate(self):
        """
        Perform one forward propagation
        """
        cache = [[self.x, self.x]]

        print("Beginning forward pass...")

        for i, v in enumerate(self.layers):
            if i == 0:
                continue
            print("Propagating from layer " + str(i-1) + " to layer " + str(i))

            # Transform the bias to the correct shape
            b_matrix = np.tile(self.params[i][1], (self.x.shape[0], 1))

            # Calculate neuron value pre activation
            neuron_outputs = np.matmul(cache[i-1][0], np.transpose(self.params[i][0])) + b_matrix

            # Activated neuron outputs
            activated_outputs = sigmoid(neuron_outputs)

            cache.append([neuron_outputs, activated_outputs])

        self.output_cache = cache

    def compute_cost(self):
        """
        Compute cost
        """
        print("Calculating cost using Mean Squared Error")
        final_activation = self.output_cache[len(self.layers)-1][1]


        # Use mean squared error loss
        mse = ((final_activation - self.y)**2).mean(axis = None)

        self.cost = mse

    def back_propagate(self):
        """
        Back propagate the error and store results
        """
        print("Backpropagating error")
        
    




eg_x = np.random.random((100,4))
eg_y = np.random.random((100,1))
test = basicNet(eg_x, eg_y, 6, sigmoid, 0.2)

test.initialise()
test.forward_propagate()
test.compute_cost()
