# Spotify Track Popularity Predictor
# Overview
This application, built using Streamlit, is designed to predict the popularity of a track on Spotify based on its musical features and genre. Leveraging the capabilities of an XGBoost regression model, it provides insights into how likely a track is to be popular among Spotify users.

# Features
Predict the Spotify popularity score of a track based on its characteristics.
Input fields for musical features like danceability, energy, loudness, tempo, and more.
Ability to select key, time signature, and genre for a more accurate prediction.
Instant prediction of the track's popularity score and its categorical interpretation (Low, Medium, High).

# How to Use
Enter Track Features: Fill in the track details such as duration, danceability, energy, loudness, and other musical attributes.
Select Key and Time Signature: Choose the track's key and time signature from the dropdown menus.
Choose Genre: Select the track's genre from a comprehensive list of genres.
Get Popularity Prediction: Click the 'Predict' button to receive a popularity score and category based on the input data.

# Technical Details
The app uses a pre-trained XGBoost model (spotify.pkl) for prediction.
Handles a mix of numerical inputs and categorical data (one-hot encoded for genre).
The prediction output is a popularity score ranging from 0 to 100, categorized into Low, Medium, or High popularity.

# Data and Model
The model was trained on a dataset comprising various Spotify tracks with their features and popularity scores.
Popularity is defined by Spotify and is based on the total number of plays the track has had and how recent those plays are.

# Disclaimer
This tool is intended for informational and exploratory purposes. It offers predictions based on the model's training data and may not reflect real-time popularity on Spotify.
