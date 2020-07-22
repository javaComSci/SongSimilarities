import pickle
import pandas as pd
from Vectorizer import Vectorizer
import numpy as np

"""
    Create the model that will be trained for the word embeddings
"""
class Model:
    def __init__(self):
        self.context_inputs = None
        self.word_outputs = None



    """
        Create the vector representation of the input data
    """
    def vectorize(self, vocab, slide_size = 2, verbose = False):
        vectorizer = Vectorizer()

        # create the vector for single words as the outputs
        one_hot_vectors = vectorizer.one_hot_vectorize(vocab)

        if verbose == True:
            print("SHAPE OF VECTORS CREATED: ", one_hot_vectors.shape)

        # get the tokens for each song
        song_artist_lyrics_df = pd.read_pickle("SongArtistLyricsTokenized.pkl")

        if verbose == True:
            print("SONG ARTIST LYRICS DATAFRAME: ")
            print(song_artist_lyrics_df.head(1))
        

        first_overall = True

        # create the input context and output word for each of the words
        for index, row in song_artist_lyrics_df.iterrows():
            
            # temp stack for specific song
            first_addition = True
            temp_inputs = None
            temp_outputs = None

            if verbose == True:
                print(index)
            
            tokens = row["Lyrics"]

            # create the sliding window for each word to obtain context
            for i in range(slide_size, len(tokens) - slide_size):
                # obtain the word and the context
                word = tokens[i]
                context = tokens[i - slide_size: i] + tokens[i + 1: i + slide_size + 1]

                # obtain the vectorized representation
                vector_output = vectorizer.get_one_hot_vector(word)
                vector_input = vectorizer.get_averager_vector(context)

                # if verbose == True:
                #     print("CONTEXT: ", context)
                #     print("INPUT: ", vector_input)
                #     print("WORD: ", word)
                #     print("OUTPUT: ", vector_output)
            
                # add to the list of inputs and outputs
                if first_addition == True:
                    first_addition = False
                    temp_inputs = vector_input
                    temp_outputs = vector_output
                else:
                    temp_inputs = np.hstack((temp_inputs, vector_input))
                    temp_outputs = np.hstack((temp_outputs, vector_output))
            

            if first_overall == True:
                first_overall = False
                self.context_inputs = temp_inputs
                self.word_outputs = temp_outputs
            else:
                self.context_inputs = np.hstack((self.context_inputs, temp_inputs))
                self.word_outputs  = np.hstack((self.word_outputs, temp_outputs))
                print("CONTEXT INPUTS SHAPE: ", self.context_inputs.shape)
                print("WORD OUTPUTS SHAPE: ", self.word_outputs.shape)
                    
        
        if verbose == True:
            print("CONTEXT INPUTS SHAPE: ", self.context_inputs.shape)
            print("WORD OUTPUTS SHAPE: ", self.word_outputs.shape)
            # print(np.nonzero(self.context_inputs[:,0]))
            # print(self.context_inputs[:,0].shape)
        

        # save the inputs and outputs as a pickle
        np.save("ContextInputs", self.context_inputs)
        np.save("WordOutputs", self.word_outputs)




if __name__ == "__main__":
    # get the data necessary to train the model
    vocab = None
    with open("Vocabulary.pkl", 'rb') as handle:
        vocab = pickle.load(handle)

    # create model 
    model = Model()

    # get the inputs and the output vectors for the model
    slide_size = 2
    model.vectorize(vocab, slide_size, True)


