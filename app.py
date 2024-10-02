
import streamlit as st
import gc
import librosa
import numpy as np
from transformers import pipeline
import language_tool_python  # For grammar checking
import difflib  # For basic pronunciation comparison

# Load ASR model
@st.cache_resource
def load_asr_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

# Load grammar checking tool
@st.cache_resource
def load_grammar_tool():
    return language_tool_python.LanguageTool('en-US')

# Function to process the audio and perform transcription
def process_audio(uploaded_file, asr_model):
    # Load audio file directly with librosa
    y, sr = librosa.load(uploaded_file, sr=16000)  # Ensuring the sampling rate is 16kHz
    
    # Check the duration of the audio
    duration = librosa.get_duration(y=y, sr=sr)
    
    # If duration is longer than 30 seconds, enable long-form transcription with timestamps
    if duration > 30:
        transcription = asr_model({"raw": y, "sampling_rate": sr}, return_timestamps=True)['text']
    else:
        transcription = asr_model({"raw": y, "sampling_rate": sr})['text']
    
    return transcription

# Function to perform grammar checking on the transcription
def check_grammar(transcription, grammar_tool):
    # Check the transcription for grammar errors
    matches = grammar_tool.check(transcription)
    corrected_text = grammar_tool.correct(transcription)
    return corrected_text, matches

# Function to analyze pronunciation (basic comparison approach)
def analyze_pronunciation(transcription, reference_text):
    # Use difflib to compare the transcription with the reference text
    diff = difflib.SequenceMatcher(None, transcription, reference_text)
    similarity = diff.ratio() * 100  # Percentage of similarity
    return similarity

# Streamlit app interface
st.title("Speech Transcription, Grammar Check, and Pronunciation Analysis")

uploaded_file = st.file_uploader("Upload a WAV or MP3 file (max 5 minutes)", type=["wav", "mp3"])
reference_text = st.text_area("Reference Text for Pronunciation Analysis (Optional)", "")

if uploaded_file:
    # Display audio player
    st.audio(uploaded_file)

    # Load ASR model and grammar checking tool
    asr_model = load_asr_model()
    grammar_tool = load_grammar_tool()

    # Process the audio and display transcription
    with st.spinner('Transcribing...'):
        transcription = process_audio(uploaded_file, asr_model)
        if transcription:
            st.subheader("Transcription:")
            st.write(transcription)

            # Check grammar and display corrected text
            st.subheader("Grammar Correction:")
            corrected_text, grammar_errors = check_grammar(transcription, grammar_tool)
            st.write(corrected_text)

            # Display grammar issues (if any)
            if grammar_errors:
                st.subheader("Grammar Issues Detected:")
                for issue in grammar_errors:
                    st.write(f"Issue: {issue.message} - Suggestion: {issue.replacements}")
            else:
                st.write("No grammar issues detected!")

            # If reference text is provided, analyze pronunciation
            if reference_text:
                pronunciation_similarity = analyze_pronunciation(transcription, reference_text)
                st.subheader("Pronunciation Analysis:")
                st.write(f"Similarity to reference text: {pronunciation_similarity:.2f}%")

    # Run garbage collection to manage memory
    gc.collect()

