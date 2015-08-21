__author__ = 'markstrefford'

# Basic neural network code from http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits

import random
import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    #Return the output of the network if the input is a
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    # Train the net using mini-batch stochastic gradient descent
    # Training data is a set of tuples (x, y) representing the training inputs and the desired outputs
    # If test data is provided then the network will be evaluated against the test data after each epoch
    # and partial progress printed out
    # NB: Useful for tracking but really slow!!
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            # Randomly shuffle the training data
            random.shuffle(training_data)
            # Now split up into mini batches
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2)".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    # Update the networks weights and biases applying gradient descent using back propogation to a single mini batch
    # The mini_batch is a list of tuples (x,y) and eta is the learning rate
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    # Return a tuple (nable_b, nable_w) representing the gradient for the cost function C_x
    # nabla_b and nable_w are the layer-by-layer lists of numpy arrays, similar to self.biases and self.weights
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Feedforward
        activation = x
        activations = [x] # List to store the activations, layer by layer
        zs = [] # List tp store all the z vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note the variable l in the loop brlow is used a little differently to the notation in the book, chap.2
        # Here, l=1 means the last layer of neurons, l=2 is the 2nd last layer
        # This is the opposite of the usual numbering scheme and the one in the book
        # and is used here to take advantage of the fact that Python can use negative indices in a list
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # Return the vector of partial derivatives
    # partical C_x / partial a for the output activations
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


# The sigmoid function (for a sigmoid neuron!)
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Derivative of the sigmoid function
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
