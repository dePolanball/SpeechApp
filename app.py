
import streamlit as st
from transformers import pipeline
import torch
import requests
import json

# Grammar checker using Gramformer (replace this with any lightweight grammar correction)
# For lightweight grammar correction without Java, we're using pre-trained models
# Install gramformer with: pip install gramformer
from gramformer import Gramformer

# Title of the app
st.title("Optimized Speech-to-Text with Grammar Check and Pronunciation Analysis")

# Initialize the grammar correction model
@st.cache_resource
def load_grammar_model():
    gf = Gramformer(models=1)  # Model=1 is for grammar correction
    return gf

grammar_model = load_grammar_model()

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file (Max 2 minutes)", type=["wav", "mp3", "ogg"])

# Progress bar
progress_bar = st.progress(0)

# Load the ASR model (Assuming you've pre-downloaded or want a model compatible with available resources)
@st.cache_resource
def load_asr_model():
    asr_model = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    return asr_model

asr_model = load_asr_model()

# Function to analyze pronunciation accuracy (dummy example for demonstration)
def analyze_pronunciation(text):
    # Placeholder function. Can be expanded with phoneme analysis if resources allow.
    # Here we simply highlight words longer than 7 letters as potentially hard to pronounce.
    hard_words = [word for word in text.split() if len(word) > 7]
    return hard_words

# Process and transcribe the uploaded audio file
def process_audio(uploaded_file, model):
    try:
        if uploaded_file is not None:
            # Read audio data
            audio_bytes = uploaded_file.read()

            # Limit file size and length
            if len(audio_bytes) > 2 * 60 * 1024 * 1024:  # Roughly equivalent to 2 minutes
                st.error("Audio file too long. Please upload a file shorter than 2 minutes.")
                return None

            # Increment progress bar
            progress_bar.progress(50)

            # Perform transcription
            transcription = model(audio_bytes, return_timestamps=False)

            # Complete progress bar
            progress_bar.progress(100)

            return transcription["text"]
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Display transcription, grammar correction, and pronunciation analysis
if uploaded_file:
    st.info("Processing the file... Please wait.")
    
    transcription = process_audio(uploaded_file, asr_model)
    
    if transcription:
        st.success("Transcription Complete:")
        st.write(transcription)
        
        # Grammar correction
        st.info("Checking grammar...")
        corrected_sentence = grammar_model.correct(transcription)
        st.write("Corrected Sentence:", corrected_sentence)
        
        # Pronunciation analysis
        st.info("Analyzing pronunciation...")
        hard_words = analyze_pronunciation(transcription)
        if hard_words:
            st.warning(f"Potentially hard-to-pronounce words: {', '.join(hard_words)}")
        else:
            st.success("No difficult words found in the transcription.")
    else:
        st.error("Failed to transcribe the audio.")

