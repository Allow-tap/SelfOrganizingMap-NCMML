import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import glob
import pandas as pd
from time import sleep


my_client_id = 'CLIENT_ID'
my_client_secret = 'CLIENT_SECRET'


sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=my_client_id,
                                                           client_secret=my_client_secret))


#get track uri
#urn syntax spotify:track:URI link
urn = 'spotify:track:4qg8VPdRnO3qLOHKt6HvT4'

track = sp.track(urn)
#print track features, on track features is displayed the artist's name
print("Track features:")
print(track)

print('End of track features:')

#audio features, audio features have the other data that is required (key, tempo, etc)
audio_features = sp.audio_features('4qg8VPdRnO3qLOHKt6HvT4')


print("Audio features:")
print(audio_features)
print('End of audio features:')

#code to retrieve the genre of a track
#urn = track['artists'][0]['uri']

#artist = sp.artist(urn)

#print(artist['genres'])