import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
from pydub import AudioSegment
import tempfile
import os

# Function to extract features from audio file
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

# Load the saved model
@st.cache_resource
def load_model_cached():
    return load_model('cat_dog_audio_model.h5')

model = load_model_cached()

st.title('Cat/Dog Audio Classifier')

uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a'])

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button('Classify'):
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_filename = tmp_file.name
                
            # Convert the uploaded file to wav format
            audio = AudioSegment.from_file(uploaded_file)
            audio.export(tmp_filename, format="wav")
            
            # Load the audio file using librosa
            audio, sr = librosa.load(tmp_filename, duration=2.5, offset=0.6)
            
            # Remove the temporary file
            os.unlink(tmp_filename)

            # Extract features
            features = extract_features(audio, sr)

            # Normalize and reshape the features
            features = features / np.max(features)
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=2)

            # Make prediction
            prediction = model.predict(features)
            class_names = ['Cat', 'Dog']
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            st.write(f"Predicted class: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")

            # Display bar chart
            st.bar_chart({class_name: pred for class_name, pred in zip(class_names, prediction[0])})

        except Exception as e:
            st.error(f"An error occurred while processing the audio file: {str(e)}")
            st.write("Please make sure you've uploaded a valid audio file (WAV, MP3, or M4A).")

st.write("Note: This model has been trained on a limited dataset and may not be accurate for all audio samples.")