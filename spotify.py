import pandas as pd
import streamlit as st
from xgboost import XGBRegressor
import joblib

# Load your trained model
model = joblib.load("spotify.pkl")

# Title and introduction
st.image('header_image.png', width=150) 
st.title("Spotify Track Popularity Predictor")
st.write("Enter the track features to predict its popularity on Spotify.")

# Input fields for all features
duration_ms = st.number_input('Duration (ms)', min_value=0)
danceability = st.number_input('Danceability', min_value=0.0, max_value=1.0, step=0.01)
energy = st.number_input('Energy', min_value=0.0, max_value=1.0, step=0.01)
loudness = st.number_input('Loudness')
mode = st.number_input('Mode', min_value=0, max_value=1)
speechiness = st.number_input('Speechiness', min_value=0.0, max_value=1.0, step=0.01)
acousticness = st.number_input('Acousticness', min_value=0.0, max_value=1.0, step=0.01)
instrumentalness = st.number_input('Instrumentalness', min_value=0.0, max_value=1.0, step=0.01)
liveness = st.number_input('Liveness', min_value=0.0, max_value=1.0, step=0.01)
valence = st.number_input('Valence', min_value=0.0, max_value=1.0, step=0.01)
tempo = st.number_input('Tempo', min_value=0.0)

key = st.selectbox('Key', options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
time_signature = st.selectbox('Time Signature', options=[0, 1, 3, 4, 5])
# List of genres
genres = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime',
          'black-metal', 'bluegrass', 'blues', 'brazil', 'breakbeat', 'british',
          'cantopop', 'chicago-house', 'children', 'chill', 'classical', 'club', 'comedy',
          'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno',
          'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro',
          'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german',
          'gospel', 'goth', 'grindcore', 'groove', 'grunge', 'guitar', 'happy',
          'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'honky-tonk',
          'house', 'idm', 'indian', 'indie-pop', 'indie', 'industrial', 'iranian',
          'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin',
          'latino', 'malay', 'mandopop', 'metal', 'metalcore', 'minimal-techno', 'mpb',
          'new-age', 'opera', 'pagode', 'party', 'piano', 'pop-film', 'pop', 'power-pop',
          'progressive-house', 'psych-rock', 'punk-rock', 'punk', 'r-n-b', 'reggae',
          'reggaeton', 'rock-n-roll', 'rock', 'rockabilly', 'romance', 'sad', 'salsa',
          'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep',
          'songwriter', 'soul', 'spanish', 'study', 'swedish', 'synth-pop', 'tango',
          'techno', 'trance', 'trip-hop', 'turkish', 'world-music']

track_genre = st.selectbox('Genre', options=genres)

# Create a DataFrame for input features
input_data = pd.DataFrame([[
    duration_ms, danceability, energy, loudness, mode, speechiness, 
    acousticness, instrumentalness, liveness, valence, tempo, key, time_signature, track_genre
]], columns=[
    'duration_ms', 'danceability', 'energy', 'loudness', 'mode', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'key', 'time_signature', 'track_genre'
])

# Apply one-hot encoding
input_data = pd.get_dummies(input_data)

# Fill missing columns with zeros
missing_cols = set(model.get_booster().feature_names) - set(input_data.columns)
for c in missing_cols:
    input_data[c] = 0

# Ensure the order of columns matches the training data
input_data = input_data[model.get_booster().feature_names]

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_data)[0]

    # Provide context about the popularity score
    st.write(f'Predicted Popularity Score: {prediction:.2f} (out of 100)')

    # Categorize the popularity
    if prediction < 20:
        popularity_category = 'Low'
    elif prediction < 60:
        popularity_category = 'Medium'
    else:
        popularity_category = 'High'

    st.write(f'Popularity Category: {popularity_category}')
