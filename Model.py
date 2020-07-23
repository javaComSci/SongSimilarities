import pickle
import pandas as pd
from Vectorizer import Vectorizer
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

"""
    Create the model that will be trained for the word embeddings
"""
class Model:
    def __init__(self, vocab_size, hidden_layer_size, batch_size, inputs, outputs):
        # sizes necessary for computation
        self.vocab_size = vocab_size
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size

        # number of examples to train on
        self.examples_size = inputs.shape[1]

        # training data for model
        self.inputs = inputs.T
        self.outputs = outputs.T

    """
        Train and create model
    """
    def train(self):
        # create sequential model with layers
        model = Sequential()

        # hidden layer that has relu activation function
        model.add(Dense(self.hidden_layer_size, activation = "relu"))

        # output layer with softmax activation to see the final outputs
        model.add(Dense(self.vocab_size, activation = "softmax"))

        # compile with categorial cross entropy loss to be minimized
        model.compile(optimizer='adam', loss = "categorical_crossentropy", metrics=["accuracy"])

        # fit the model with given inputs and outputs
        model.fit(self.inputs, self.outputs, epochs = 150, batch_size = self.batch_size)

        # evaluate the accuracy of the model
        x, accuracy = model.evaluate(self.inputs, self.outputs)
        
        # predict with the actual inputs
        print(model.predict(self.inputs))

        model.save("TrainedModel")
    


    """
        Continue training model based on the trained model left off
    """
    def continue_train(self):
        # load model for more training
        model = keras.models.load_model("TrainedModel")

        # continue fit and evaluate
        model.fit(self.inputs, self.outputs, epochs = 150, batch_size = self.batch_size)
        x, accuracy = model.evaluate(self.inputs, self.outputs)
        print(model.predict(self.inputs))

        # save updated model
        model.save("TrainedModel")

    

    """
        Extract the weights of the model as the word embeddings weights
    """
    def extract_weights(self):
        # load model for extracting weights
        model = keras.models.load_model("TrainedModel")

        # get the weights between each layer
        weights = model.get_weights()

        # put into numpy arrays - need to get indicies 0 and 2 since the bias vectors are arrays 1 and 3
        self.weights_1 = np.array(weights[0]).reshape((self.vocab_size, self.hidden_layer_size))
        self.weights_2 = np.array(weights[2]).reshape((self.hidden_layer_size, self.vocab_size))

        # print(self.weights_1.shape)
        # print(self.weights_2.shape)
            
        # model.summary()
    


    """
        Average weight vectors to create embeddings for all the words
    """
    def set_embeddings(self):
        self.embeddings = (self.weights_1 + self.weights_2.T)/2.0

        np.save("WordEmbeddings", self.embeddings)
        # print(self.embeddings.shape)

    

if __name__ == "__main__":
    # create the model with given parameters
    vocab_size = 4350
    hidden_layer_size = 50
    batch_size = 120

    # obtain the inputs and outputs from the saved numpy files
    inputs = np.load("ContextInputs.npy")
    outputs = np.load("WordOutputs.npy")
    
    model = Model(vocab_size, hidden_layer_size, batch_size, inputs, outputs)

    model.train()

    # model.continue_train()

    model.extract_weights()

    model.set_embeddings()


