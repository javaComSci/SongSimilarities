import pickle
import pandas as pd
from Vectorizer import Vectorizer


"""
    Create the model that will be trained for the word embeddings
"""
class Model:
    def __init__(self):
        self.vectorizer = Vectorizer()
        self.inputs = []
        self.outputs = []


    """
        Create the vector representation of the input data
    """
    def vectorize(self, vocab, slide_size = 2):
        # create the vector for single words
        one_hot_vectors = self.vectorizer.one_hot_vectorize(vocab)
        

        print(one_hot_vectors)




if __name__ == "__main__":
    # get the data necessary to train the model
    vocab = None
    with open("Vocabulary.pkl", 'rb') as handle:
        vocab = pickle.load(handle)

    # create model 
    model = Model()

    # get the inputs and the output vectors for the model
    slide_size = 2
    model.vectorize(vocab, slide_size)


