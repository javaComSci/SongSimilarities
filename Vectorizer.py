import pickle
import pandas as pd


"""
    Vectorizing class in order to change the tokens into numeric vectors
"""
class Vectorizer():
    def __init__(self):
        self.wordToIndex = {}
        self.indexToWord = {}
        self.one_hot_vectors = []


    """
        Apply one hot vectorization for the tokens
    """
    def one_hot_vectorize(self, vocab):
        # create the one hot vectors
        self.one_hot_vectors = [[1 if j == i else 0 for j in range(len(vocab))] for i in range(len(vocab))]
        
        # create a dictionary to keep track of the words that correspond to indicies
        for index, word in enumerate(vocab):
            self.wordToIndex[word] = index
            self.indexToWord[index] = word

        return self.one_hot_vectors