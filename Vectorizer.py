import pickle
import pandas as pd
import numpy as np

"""
    Vectorizing class in order to change the tokens into numeric vectors
"""
class Vectorizer():
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.one_hot_vectors = None
        self.vocab_size = 0


    """
        Apply one hot vectorization for all tokens
    """
    def one_hot_vectorize(self, vocab):
        # create the one hot vectors
        self.one_hot_vectors = [[1 if j == i else 0 for j in range(len(vocab))] for i in range(len(vocab))]
        self.one_hot_vectors = np.array(self.one_hot_vectors).T
        self.vocab_size = self.one_hot_vectors.shape[0]

        # create a dictionary to keep track of the words that correspond to indicies
        for index, word in enumerate(vocab):
            self.word_to_index[word] = index
            self.index_to_word[index] = word

        return self.one_hot_vectors
    


    """
        Get the one hot vectorization for a specific token
    """
    def get_one_hot_vector(self, token):
        if token in self.word_to_index:
            return self.one_hot_vectors[self.word_to_index[token]].reshape(self.vocab_size, 1)
        else:
            print("NOT A VALID TOKEN")
            return np.zeros((self.vocab_size, 1))

    

    """
        Get the averager vector given the context
    """
    def get_averager_vector(self, context):
        # create a column vector to contain the averages
        vocab_size = self.one_hot_vectors.shape[0]
        total = np.zeros((self.vocab_size, 1))

        # add up all the one hot vectors of the tokens
        for token in context:
            token_vector = self.get_one_hot_vector(token)
            total += token_vector
        
        # create the average of the embeddings in the context
        average = total/len(context)
        return average