# SongSimilarities
There are many songs and artists in our world today. In these songs, it would be interesting to find the songs and artists that are similar and different
with the use of natural language processing techniques. In order to do that, the goal of this application is to create a CBOW model in order to extract word embeddings. These word embeddings will be further used
to find the meanings of the words in the songs. With this information, vector similarity measures such as cosine similarity can be used in order to find 
songs with similar meanings to find out which songs a person could like based on the songs and artists they already like. 

## Goals
1. Create vector representation of words in the vocabulary
2. Train a CBOW model that takes in context and outputs the words in order to create weight matricies
3. Extract weight matricies in order to obtain the representation of words
4. Create representation of songs based on the words
5. Measure meaning of songs with vector similarity measures such as cosine similarity
