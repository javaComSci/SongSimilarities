import pickle
import pandas as pd
from Vectorizer import Vectorizer
import numpy as np

"""
    Create the model that will be trained for the word embeddings
"""
class Model:
    def __init__(self, vocab_size, hidden_layers, batch_size, inputs, outputs):
        # sizes necessary for computation
        self.vocab_size = vocab_size
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size

        # number of examples to train on
        self.examples_size = inputs.shape[1]

        # training data for model
        self.inputs = inputs
        self.outputs = outputs

        # vectors necessary for computation
        self.weights_1 = None
        self.weights_2 = None
        self.bias_1 = None
        self.bias_2 = None
    


    """
        Creates the relu function for computation
    """
    def relu(self, m):
        # print(m)
        return np.maximum(0, m)
    


    """
        Creates the softmax function for computation
    """
    def softmax(self, m):
        return np.exp(m)/np.sum(np.exp(m), axis = 0)



    """
        Initialize all the matricies and vectors in order to train the model
    """
    def initialize(self):
        # print("INITIALIZATION")

        # input layer to hidden layer
        self.weights_1 = np.random.rand(self.hidden_layers, self.vocab_size)
        self.bias_1 = np.random.rand(self.hidden_layers, 1)
    
        # hidden layer to output layer
        self.weights_2 = np.random.rand(self.vocab_size, self.hidden_layers)
        self.bias_2 = np.random.rand(self.vocab_size, 1)



    """
        Compute the cost of using the current weights
    """
    def get_cost(self, predictions, batch_outputs):

        # check for shape matching
        assert (predictions.shape == batch_outputs.shape), "Predictions and outputs do not match"
        
        # calculate the cost with cross entropy
        logprobs = np.multiply(np.log(predictions),batch_outputs) + np.multiply(np.log(1 - predictions), 1 - batch_outputs)
        cost = - 1/self.batch_size * np.sum(logprobs)
        cost = np.squeeze(cost)

        return cost



    """
        Calculate the value of the prediction given the inputs and weights
    """
    def forward_propogate(self, batch_inputs):
        # print("FORWARD PROPOGATE")

        # calculate relu(W1x + b1)
        weights_1_vocab_bias = self.relu(np.dot(self.weights_1, batch_inputs) + self.bias_1)
        assert (weights_1_vocab_bias.shape[0] == self.hidden_layers), "Addition of weights and bias incorrect 1"

        # calculate softmax(W2*relu(W1x + b1) + b2)
        weights_2_vocab_bias = self.softmax(np.dot(self.weights_2, weights_1_vocab_bias) + self.bias_2)
        assert (weights_2_vocab_bias.shape[0] == self.vocab_size), "Addition of weights and bias incorrect 2"

        return weights_2_vocab_bias, weights_1_vocab_bias



    """
        Calculate the updates to the weights and bias vectors
    """
    def backward_propogate(self, batch_inputs, batch_outputs, predictions, relu_results, alpha = 0.05):
        # print("BACKWARD PROPOGATE")

        # compute the necessary gradients for each of the weights and biases
        l1 = np.dot(self.weights_2.T, predictions - batch_outputs)
        l1 = np.maximum(l1, 0)
        grad_W1 = np.dot(l1, batch_inputs.T)/self.batch_size
        grad_W2 = np.dot(predictions - batch_outputs, relu_results.T)/self.batch_size
        grad_b1 = np.dot(l1, np.ones((self.batch_size, 1)))/self.batch_size
        grad_b2 = np.dot(predictions - batch_outputs, np.ones((self.batch_size, 1)))/self.batch_size
    
        assert(self.weights_1.shape == grad_W1.shape), "Weights and gradients shape not same 1"
        assert(self.weights_2.shape == grad_W2.shape), "Weights and gradients shape not same 2"
        assert(self.bias_1.shape == grad_b1.shape), "Bias and gradients shape not same 1"
        assert(self.bias_2.shape == grad_b2.shape), "Bias and gradients shape not same 2"

        # update the bias and weight vectors
        self.weights_1 = self.weights_1 - (alpha * grad_W1) 
        self.weights_2 = self.weights_2 - (alpha * grad_W2)
        self.bias_1 = self.bias_1 - (alpha * grad_b1)
        self.bias_2 = self.bias_2 - (alpha * grad_b2)


    
    """
        Evaluate the model with the current weights and outputs
    """
    def evaluate(self):

        # get the predictions
        predictions, relu_results = self.forward_propogate(inputs)

        # get the actual predictions
        max_columns_predictions = np.argmax(predictions, axis=0)

        # get the best columns in the outputs
        max_columns_outputs = np.argmax(outputs, axis = 0)
        
        assert(max_columns_outputs.shape == max_columns_predictions.shape), "Outputs and predictons not same length"
        print("SHAPE IS: ", max_columns_predictions.shape[0])

        # get the correct ones
        correct = 0
        for i in range(max_columns_predictions.shape[0]):
            print(max_columns_predictions[i], max_columns_outputs[i])
            if max_columns_predictions[i] == max_columns_outputs[i]:
                correct += 1

        # calculate the accuracy
        accuracy = (correct/max_columns_predictions.shape[0])
        print("ACCURACY", accuracy)



    """
        Train the model with the inputs and outputs
    """
    def train(self, times_to_repeat):
        alpha = 0.05
        for j in range(times_to_repeat):

            self.evaluate()

            # reduce the learning rate
            alpha = alpha * 0.5

            # train by applying forward and backward propogation
            for i in range(0, self.examples_size - self.batch_size, self.batch_size):

                # get current inputs and outputs to train on
                batch_inputs = self.inputs[:,i: i + batch_size]
                batch_outputs = self.outputs[:, i : i + batch_size]

                # get the predictions through forward propogation
                predictions, relu_results = self.forward_propogate(batch_inputs)
                

                # display the cost
                if i % 1000 == 0:
                    cost = self.get_cost(predictions, batch_outputs)
                    print("ITERATIONS: ", i)
                    print("COST: ", cost)
                
                # do back propogation to update the weights
                self.backward_propogate(batch_inputs, batch_outputs, predictions, relu_results, alpha)
        



if __name__ == "__main__":
    # obtain the inputs and outputs from the saved numpy files
    inputs = np.load("ContextInputs.npy")
    outputs = np.load("WordOutputs.npy")

    # create the model with given parameters
    vocab_size = 4350
    hidden_layers = 2
    batch_size = 50

    # create a model
    model = Model(vocab_size, hidden_layers, batch_size, inputs, outputs)

    # initialize all the matricies for computation
    model.initialize()

    # train the model
    times_to_repeat = 5
    model.train(times_to_repeat)

    # evauate the model
    model.evaluate()