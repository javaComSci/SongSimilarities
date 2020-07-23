import pickle
import pandas as pd
import numpy as np


"""
    Similarity checker class that checks songs that are similar to each other
"""
class SimilarityChecker:
    def __init__(self, word_embeddings, word_to_index, index_to_word, song_artist_tokenized):
        self.word_embeddings = word_embeddings
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.song_artist_tokenized = song_artist_tokenized

        # embed all the songs that have been tokenized
        self.embed_songs()



    """
        Embed all the songs with the words
    """
    def embed_songs(self):
        # dictionary with the embeddings
        self.song_artist_to_embeddings = {}
        self.artist_to_embeddings = {}
        self.artist_song_counts = {}

        # calculate for every song
        for index, row in self.song_artist_tokenized.iterrows():

            # get all the tokens of the song and create an array for embeddings
            tokens = row["Lyrics"]
            song_artist = row["Song_Artist"]
            artist = song_artist[1]
            embedding_value = np.zeros((1, 50))
            valid_words_counts = 0

            # check the token validity and add embedding of token
            for token in tokens:
                if token in self.word_to_index:
                    index_of_word = self.word_to_index[token]
                    embeddings = word_embeddings[index][:]

                    # for specific songs by artists
                    embedding_value += embeddings
                    valid_words_counts += 1

            # for artists
            if artist in self.artist_to_embeddings:
                self.artist_to_embeddings[artist] += embedding_value
                self.artist_song_counts[artist] += valid_words_counts
            else:
                self.artist_to_embeddings[artist] = np.zeros((1, 50))
                self.artist_to_embeddings[artist] += embedding_value
                self.artist_song_counts[artist] = valid_words_counts

            # take the average of the embeddings of the sum
            embedding_value = embedding_value/valid_words_counts

            # place the embedding in the dictionary
            self.song_artist_to_embeddings[song_artist] = embedding_value
        

        # take the average for each artist
        for artist in self.artist_to_embeddings.keys():
            self.artist_to_embeddings[artist] /= self.artist_song_counts[artist]
        
        print(self.song_artist_to_embeddings.keys())
    


    """
        Find the similarity between vectors with cosine similarity
    """
    def cosine_similarity(self, embeddings, a, b):
        # get the embeddings of the songs
        a_embedding = embeddings[a]
        b_embedding = embeddings[b]

        # get the norms of the embeddings
        a_norm = np.linalg.norm(a_embedding)
        b_norm = np.linalg.norm(b_embedding)

        similarity = np.dot(a_embedding, b_embedding.T)/(a_norm * b_norm)

        print(a, b, "SIMILARITY IS ", similarity)

        return similarity



if __name__ == "__main__":
    # load up necessary information to get word information
    word_embeddings = np.load("WordEmbeddings.npy")
    
    with open("WordToIndex.pickle", "rb") as handle:
        word_to_index = pickle.load(handle)
    
    with open("IndexToWord.pickle", "rb") as handle:
        index_to_word = pickle.load(handle)
    
    song_artist_tokenized = pd.read_pickle("SongArtistLyricsTokenized.pkl")

    # create an instance to check the simliarities between songs
    similarity_checker = SimilarityChecker(word_embeddings, word_to_index, index_to_word, song_artist_tokenized)

    # similarity_checker.calculate_overall_similarities()

    a = ('Water Under the Bridge', 'Adele')
    b = ('Barcelona', 'Ed Sheeran')
    similarity_checker.cosine_similarity(similarity_checker.song_artist_to_embeddings, a, b)
