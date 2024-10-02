
import streamlit as st
import torch
import torchaudio
from transformers import pipeline

# Title of the app
st.title("Optimized Speech-to-Text with Grammar Check and Pronunciation Analysis")

# Load the grammar correction model using Hugging Face Transformers
@st.cache_resource
def load_grammar_model():
    grammar_corrector = pipeline('text2text-generation', model="prithivida/grammar_error_correcter_v1")
    return grammar_corrector

grammar_model = load_grammar_model()

# Function to load audio using torchaudio (supports MP3 and WAV)
def load_audio(file):
    try:
        waveform, sample_rate = torchaudio.load(file)
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0).unsqueeze(0)
        return waveform, sample_rate
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, None

# Progress bar
progress_bar = st.progress(0)

# Load the ASR model (wav2vec2 without the need for ffmpeg)
@st.cache_resource
def load_asr_model():
    asr_model = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    return asr_model

asr_model = load_asr_model()

# Function to analyze pronunciation accuracy (dummy example for demonstration)
def analyze_pronunciation(text):
    hard_words = [word for word in text.split() if len(word) > 7]
    return hard_words

# Upload audio file (support for both WAV and MP3)
uploaded_file = st.file_uploader("Choose an audio file (Max 2 minutes, MP3 or WAV format)", type=["wav", "mp3"])

# Process and transcribe the uploaded audio file
def process_audio(uploaded_file, model):
    try:
        if uploaded_file is not None:
            # Read the raw waveform and sample rate from the uploaded file
            waveform, sample_rate = load_audio(uploaded_file)

            if waveform is None:
                return None

            # Check file size
            if waveform.size(1) > 16000 * 120:  # Limit to 2 minutes for a sample rate of 16k
                st.error("Audio file too long. Please upload a file shorter than 2 minutes.")
                return None

            # Increment progress bar
            progress_bar.progress(50)

            # Perform transcription
            transcription = model(waveform.numpy(), return_timestamps='word')

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
        corrected_sentence = grammar_model(transcription)[0]['generated_text']
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

