import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import io
import os

# Constants (make sure these match your model's training parameters)
SAMPLE_RATE = 22050
DURATION = 5
N_FFT = 2048
HOP_LENGTH = 512

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .divider {
        height: 100%;
        border-left: 2px solid #bbb;
        margin-left: 20px;
        margin-right: 20px;
        color: black
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'chart_type' not in st.session_state:
    st.session_state.chart_type = None
if 'results' not in st.session_state:
    st.session_state.results = []

@st.cache_resource
def load_model_cached():
    model_path = 'cnn_with_fft.keras'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the model file is in the correct location.")
        st.stop()
    return load_model(model_path)

def load_and_preprocess_audio(audio_file, sr=SAMPLE_RATE, duration=DURATION):
    audio_bytes = audio_file.read()
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, duration=duration)
    target_length = duration * sr
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, max(0, target_length - len(audio))))
    fft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    fft_input = fft.reshape(1, fft.shape[0], fft.shape[1], 1)
    return audio, fft, fft_input

def create_fft_plot(results, plot_type, is_log=False):
    fig = go.Figure()
    for result in results:
        fft_simple = np.abs(np.fft.fft(result['raw_audio']))
        freqs = np.fft.fftfreq(len(result['raw_audio']), 1/SAMPLE_RATE)
        y_values = np.log(fft_simple[:len(fft_simple)//2] + 1e-10) if is_log else fft_simple[:len(fft_simple)//2]
        fig.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=y_values, mode='lines', name=result['sound']))

    fig.update_layout(
        title=f'{"Log" if is_log else "Linear"} FFT {plot_type}',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Log Magnitude' if is_log else 'Magnitude',
        height=400
    )
    return fig

def create_spectrogram(fft):
    return go.Figure(data=go.Heatmap(
        z=librosa.amplitude_to_db(fft, ref=np.max),
        colorscale='Viridis'
    )).update_layout(
        title='Spectrogram',
        xaxis_title='Time',
        yaxis_title='Frequency',
        height=300
    )

def create_waveform(raw_audio):
    times = np.linspace(0, DURATION, len(raw_audio))
    return go.Figure(data=go.Scatter(x=times, y=raw_audio, mode='lines')).update_layout(
        title='Audio Waveform',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        height=300
    )

def main():
    st.title('Cat/Dog Audio Classifier')

    model = load_model_cached()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sound 1")
        uploaded_file1 = st.file_uploader("Choose Sound 1", type=['wav', 'mp3', 'm4a'], key="file1")
        if uploaded_file1:
            st.audio(uploaded_file1)

    with col2:
        st.subheader("Sound 2")
        uploaded_file2 = st.file_uploader("Choose Sound 2", type=['wav', 'mp3', 'm4a'], key="file2")
        if uploaded_file2:
            st.audio(uploaded_file2)

    if st.button('Classify'):
        st.session_state.results = []
        for i, file in enumerate([uploaded_file1, uploaded_file2], 1):
            if file is not None:
                try:
                    raw_audio, fft, processed_audio = load_and_preprocess_audio(file)
                    prediction = model.predict(processed_audio)
                    class_names = ['Dog', 'Cat']
                    predicted_class = class_names[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100
                    st.session_state.results.append({
                        'sound': f'Sound {i}',
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'raw_audio': raw_audio,
                        'fft': fft
                    })
                except Exception as e:
                    st.error(f"An error occurred while processing Sound {i}: {str(e)}")
                    st.write(f"Please make sure you've uploaded a valid audio file for Sound {i} (WAV, MP3, or M4A).")

    # Display results side by side
    if st.session_state.results:
        st.subheader("Classification Results")
        col1, col2 = st.columns(2)
        for i, result in enumerate(st.session_state.results):
            with col1 if i == 0 else col2:
                st.markdown(f"<h4>{result['sound']}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h5>Predicted class: <strong style='color: green;'>{result['predicted_class']}</strong></h5>", unsafe_allow_html=True)
                st.write(f"Accuracy: {result['confidence']:.2f}%")

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.session_state.chart_type = st.selectbox(
                "Select Chart Type",
                ['Overlay', 'Separate'],
                index=0  # default selection
            )

        if st.session_state.chart_type == 'Overlay':
            st.subheader("Overlay FFT Graphs")
            st.plotly_chart(create_fft_plot(st.session_state.results, "Overlay", is_log=False), use_container_width=True)
            st.plotly_chart(create_fft_plot(st.session_state.results, "Overlay", is_log=True), use_container_width=True)

        elif st.session_state.chart_type == 'Separate':
            st.subheader("Individual Graphs")

            # Ensure we have results for both Sound 1 and Sound 2
            if len(st.session_state.results) == 2:
                # Create two columns with a divider in between
                col1, divider, col2 = st.columns([1, 0.02, 1])

                # For Sound 1 (First Column)
                with col1:
                    result = st.session_state.results[0]
                    st.markdown(f"<h3>Graphs for {result['sound']}</h3>", unsafe_allow_html=True)
                    linear_fft = create_fft_plot([result], "Individual", is_log=False)
                    log_fft = create_fft_plot([result], "Individual", is_log=True)
                    spectrogram = create_spectrogram(result['fft'])
                    waveform = create_waveform(result['raw_audio'])

                    st.plotly_chart(linear_fft, use_container_width=True)
                    st.plotly_chart(log_fft, use_container_width=True)
                    st.plotly_chart(spectrogram, use_container_width=True)
                    st.plotly_chart(waveform, use_container_width=True)


                with divider:
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # Insert a vertical divider
                with col2:
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # For Sound 2 (Second Column)
                with col2:
                    result = st.session_state.results[1]
                    st.markdown(f"<h3>Graphs for {result['sound']}</h3>", unsafe_allow_html=True)
                    linear_fft = create_fft_plot([result], "Individual", is_log=False)
                    log_fft = create_fft_plot([result], "Individual", is_log=True)
                    spectrogram = create_spectrogram(result['fft'])
                    waveform = create_waveform(result['raw_audio'])

                    st.plotly_chart(linear_fft, use_container_width=True)
                    st.plotly_chart(log_fft, use_container_width=True)
                    st.plotly_chart(spectrogram, use_container_width=True)
                    st.plotly_chart(waveform, use_container_width=True)
            else:
                st.write("Please upload two audio files to compare results.")




if __name__ == "__main__":
    main()
