
import streamlit as st
import gc
import librosa
import numpy as np
from transformers import pipeline

# Load ASR model
@st.cache_resource
def load_asr_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

# Function to process the audio and perform transcription
def process_audio(uploaded_file, asr_model):
    # Load audio file directly with librosa
    y, sr = librosa.load(uploaded_file, sr=16000)  # Ensuring the sampling rate is 16kHz
    
    # Check if the audio duration exceeds 5 minutes (300 seconds)
    if librosa.get_duration(y=y, sr=sr) > 300:
        st.error("The audio file exceeds the 5-minute limit. Please upload a shorter file.")
        return None

    # Perform transcription
    transcription = asr_model({"raw": y, "sampling_rate": sr})['text']
    
    return transcription

# Streamlit app interface
st.title("Lean Speech Transcription App")
uploaded_file = st.file_uploader("Upload a WAV or MP3 file (max 5 minutes)", type=["wav", "mp3"])

if uploaded_file:
    # Display audio player
    st.audio(uploaded_file)

    # Load ASR model
    asr_model = load_asr_model()

    # Process the audio and display transcription
    with st.spinner('Transcribing...'):
        transcription = process_audio(uploaded_file, asr_model)
        if transcription:
            st.write(transcription)

    # Run garbage collection to manage memory
    gc.collect()

