import pickle
import pandas as pd
import better_profanity
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt')

"""
    Convert the dictinoary into dataframe
"""
def convert_to_df(verbose = False):
    # open as pickle and change to dataframe for easy use
    with open("SongArtistLyrics.pickle", "rb") as handle:
        data_dict = pickle.load(handle)
        df = pd.DataFrame(list(data_dict.items()),columns = ["Song_Artist","Lyrics"])
        if verbose == True:
            print(df)
        return df



"""
    Clean the text in lyrics
"""
def clean_text(text, verbose = False):
    # clean text and replace with spaces
    text_new = better_profanity.profanity.censor(text)
    text_new = text_new.replace("\n", " ")
    return text_new



"""
    Clean the data in the dataframe and save it as another pickle
"""
def clean_data(df, verbose = False):
    # change to lowercase and take care of punctuation
    df["Lyrics"] = df["Lyrics"].str.lower()
    df["Lyrics"] = df["Lyrics"].str.replace("\n", " ")
    df["Lyrics"] = df["Lyrics"].str.replace(",", ".")

    # remove all bad language from songs
    better_profanity.profanity.load_censor_words()
    df["Lyrics"] = df.apply(lambda row : clean_text(row["Lyrics"]), axis = 1)
    
    if verbose == True:
        print(df)
    
    return df



"""
    Save df as pickle
"""
def save_df(df, name):
    df.to_pickle(name)



""" 
    Only keep the valid tokens
"""
def validate_token(token):
    new_token = [t for t in token if t.isalpha() or t == "."]
    return new_token



"""
    Tokenize all the words in the songs
"""
def tokenize_lyrics(df, verbose = False):
    # tokenize all the lines
    df["Lyrics"] = df.apply(lambda row : word_tokenize(row["Lyrics"]), axis = 1)
    # only keep the valid tokens
    df["Lyrics"] = df.apply(lambda row : validate_token(row["Lyrics"]), axis = 1)
    
    if verbose == True:
        print(df)
    
    return df



if __name__ == "__main__":
    df = convert_to_df(False)
    df = clean_data(df, False)
    save_df(df, "SongArtistLyricsCleaned.pkl")
    df = tokenize_lyrics(df, False)
    save_df(df, "SongArtistLyricsTokenized.pkl")
