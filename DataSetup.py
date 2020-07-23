import lyricsgenius
import pickle
import os


"""
    Get the lyrics of songs and place in a pickle from API
"""
def get_artists_songs(genius, verbose = False):
    # list of all the artists to get song lyrics from
    artists_to_fetch = ["Ed Sheeran", "Justin Bieber", "Katy Perry", "Jennifer Lopez", "Adele", "Coldplay", "Drake", "Maroon 5", "The Proclaimers"]

    # dictionary to get pickled
    song_artist_to_lyrics = {}

    # get the song lyrics from each of the artists popular songs
    for artist in artists_to_fetch:
        if verbose == True:
            print("ARTIST: ", artist)
        artist_infos = genius.search_artist(artist, max_songs=20, sort="popularity")
        if artist_infos != None:
            for song in artist_infos.songs:
                if verbose == True:
                    print("SONG: ", song)
                song_info = genius.search_song(song.title, artist_infos.name)
                if song_info != None:
                    song_artist = (song.title, artist)
                    song_artist_to_lyrics[song_artist] = song_info.lyrics
        
    
    # pickle the information for later use
    with open("SongArtistLyrics.pickle", "wb") as handle:
        # pickle.dump({"testing1": data}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(song_artist_to_lyrics, handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    GENIUS_ACCESS_TOKEN = os.environ.get("GENIUS_CLIENT_ACCESS_TOKEN")
    genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)
    genius.remove_section_headers = True 
    genius.verbose = False

    get_artists_songs(genius, True)
    
