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
        self.context_inputs = None
        self.word_outputs = None


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

    

    """
        Create the vector representation of the input data
    """
    def vectorize(self, vocab, slide_size = 2, verbose = False):

        # create the vector for single words as the outputs
        one_hot_vectors = self.one_hot_vectorize(vocab)

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
                vector_output = self.get_one_hot_vector(word)
                vector_input = self.get_averager_vector(context)

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

    # create vectorizer 
    vectorizer = Vectorizer()

    # get the inputs and the output vectors for the model
    slide_size = 2
    vectorizer.vectorize(vocab, slide_size, True)
