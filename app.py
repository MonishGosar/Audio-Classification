import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import io
from pydub import AudioSegment
import tempfile
import os
import traceback

# Constants (make sure these match your model's training parameters)
SAMPLE_RATE = 22050
DURATION = 5
N_FFT = 2048
HOP_LENGTH = 512
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

st.set_page_config(layout="wide")

@st.cache_resource
def load_model_cached():
    model_path = 'cnn_with_fft.keras'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    return load_model(model_path)

def load_and_preprocess_audio(audio_file, sr=SAMPLE_RATE, duration=DURATION):
    # Load audio file with a fixed duration
    audio, _ = librosa.load(audio_file, sr=sr, duration=duration)
    
    # Pad or truncate the audio to the fixed duration
    target_length = duration * sr
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, max(0, target_length - len(audio))))
    
    # Compute FFT
    fft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    
    # Reshape for model input
    fft_input = fft.reshape(1, fft.shape[0], fft.shape[1], 1)
    
    return audio, fft, fft_input

st.title('Cat/Dog Audio Classifier')

model = load_model_cached()

if model is None:
    st.error("Failed to load the model. Please check if the model file exists and is accessible.")
    st.stop()

uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a'])

if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)
    
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File is too large. Please upload a file smaller than {MAX_FILE_SIZE/1024/1024}MB.")
    else:
        st.audio(uploaded_file)
        
        if st.button('Classify'):
            try:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_filename = tmp_file.name
                    # Write the uploaded file to the temporary file
                    tmp_file.write(uploaded_file.getvalue())
                
                # Convert the uploaded file to wav format
                audio = AudioSegment.from_file(tmp_filename)
                audio.export(tmp_filename, format="wav")
                
                # Load and preprocess the audio
                raw_audio, fft, processed_audio = load_and_preprocess_audio(tmp_filename)
                
                # Remove the temporary file
                os.unlink(tmp_filename)

                # Make prediction
                prediction = model.predict(processed_audio)
                class_names = ['Dog', 'Cat']
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                st.write(f"Predicted class: {predicted_class}")
                st.write(f"Confidence: {confidence:.2f}%")

                # Create two columns for FFT and Spectrogram
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Simple FFT")
                    # Compute and plot simple FFT
                    fft_simple = np.abs(np.fft.fft(raw_audio))
                    freqs = np.fft.fftfreq(len(raw_audio), 1/SAMPLE_RATE)
                    
                    fig_fft = go.Figure(data=go.Scatter(x=freqs[:len(freqs)//2], y=fft_simple[:len(fft_simple)//2], mode='lines'))
                    fig_fft.update_layout(
                        title='FFT',
                        xaxis_title='Frequency (Hz)',
                        yaxis_title='Magnitude',
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=300,
                        width=100
                    )
                    st.plotly_chart(fig_fft, use_container_width=True)

                with col2:
                    st.subheader("Spectrogram")
                    # Create spectrogram plot
                    fig_spec = go.Figure(data=go.Heatmap(
                        z=librosa.amplitude_to_db(fft, ref=np.max),
                        colorscale='Viridis'
                    ))
                    fig_spec.update_layout(
                        title='Spectrogram',
                        xaxis_title='Time',
                        yaxis_title='Frequency',
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=300,
                        width=None  # Let Streamlit decide the width
                    )
                    st.plotly_chart(fig_spec, use_container_width=True)

                st.subheader("Audio Waveform")
                # Plot waveform
                times = np.linspace(0, DURATION, len(raw_audio))
                fig_wave = go.Figure(data=go.Scatter(x=times, y=raw_audio, mode='lines'))
                fig_wave.update_layout(
                    title='Audio Waveform',
                    xaxis_title='Time (s)',
                    yaxis_title='Amplitude',
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=300,
                    width=None  # Let Streamlit decide the width
                )
                st.plotly_chart(fig_wave, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred while processing the audio file: {str(e)}")
                st.text(traceback.format_exc())
                st.write("Please make sure you've uploaded a valid audio file (WAV, MP3, or M4A).")

