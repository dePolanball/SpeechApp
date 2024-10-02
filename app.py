
import streamlit as st
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import requests
import librosa
import numpy as np
import gc

# Function to clear cache and reduce memory usage
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

# Load models for transcription and phoneme analysis, but only when needed
@st.cache_resource
def load_phoneme_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

@st.cache_resource
def load_asr_model():
    # Whisper model for smaller, faster processing
    asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    return asr_model

# Grammar check using LanguageTool API
def grammar_check(text):
    api_url = "https://api.languagetool.org/v2/check"
    params = {
        'text': text,
        'language': 'en-US'
    }
    response = requests.post(api_url, data=params)
    matches = response.json().get("matches", [])
    
    return matches

# Function to chunk audio into smaller parts
def chunk_audio(y, sr, chunk_duration=30):
    chunk_length = int(chunk_duration * sr)  # chunk_duration in seconds
    chunks = [y[i:i + chunk_length] for i in range(0, len(y), chunk_length)]
    return chunks

# Transcribe chunks of audio with progress bar
def transcribe_chunks(chunks, sr, model):
    transcription = []
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Transcribing chunk {i+1}/{len(chunks)}..."):
            inputs = {"raw": chunk, "sampling_rate": sr}
            result = model(inputs)
            transcription.append(result['text'])
        
        # Update the progress bar after each chunk
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)
    
    return " ".join(transcription)

# Load the models
processor, phoneme_model = load_phoneme_model()
asr_model = load_asr_model()

# Streamlit app interface
st.title("Optimized Speech Transcription and Phoneme Analysis v 0.0.2")

st.write("Upload an audio file (WAV, MP3, OGG, under 5 minutes) for optimized analysis.")

# Restrict file uploader to audio files of a reasonable size
uploaded_file = st.file_uploader("Choose an audio file (max 5 minutes)", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Play the uploaded audio file
    st.audio(uploaded_file, format=uploaded_file.type)

    # Load the audio file with correct sampling rate (16 kHz) for compatibility with the models
    try:
        y, sr = librosa.load(uploaded_file, sr=16000)  # Ensure the sampling rate is 16 kHz
        duration_minutes = librosa.get_duration(y=y, sr=sr) / 60
        if duration_minutes > 5:
            st.error("The uploaded audio file is longer than 5 minutes. Please upload a shorter file.")
            clear_cache()
        else:
            # Chunk the audio into 30-second segments
            chunks = chunk_audio(y, sr, chunk_duration=30)

            # Transcribe each chunk and merge results, with progress bar
            transcription = transcribe_chunks(chunks, sr, asr_model)
            st.subheader("Transcription:")
            st.write(transcription)

            # Phoneme analysis using Wav2Vec2 for pronunciation evaluation
            with st.spinner("Analyzing pronunciation..."):
                input_values = processor(y, return_tensors="pt", sampling_rate=sr).input_values
                logits = phoneme_model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_phonemes = processor.batch_decode(predicted_ids)

                # Simplistic pronunciation scoring based on the number of recognizable phonemes
                phoneme_score = np.mean([phoneme != '' for phoneme in predicted_phonemes[0]])
                pronunciation_score = phoneme_score * 100  # Convert to percentage

                st.subheader("Pronunciation Evaluation:")
                st.write(f"Pronunciation Score: {pronunciation_score:.2f} / 100")
                st.write("Recognized Phonemes: ", predicted_phonemes[0])

            # Grammar evaluation using LanguageTool API
            with st.spinner("Checking grammar..."):
                grammar_issues = grammar_check(transcription)
                grammar_score = max(0, 100 - len(grammar_issues) * 5)  # Deduct points per issue
                st.subheader("Grammar Evaluation:")
                st.write(f"Grammar Score: {grammar_score:.2f} / 100")
                st.write("Grammar Issues:")
                for issue in grammar_issues:
                    st.write(f"- {issue['message']}")
    except Exception as e:
        st.error(f"Error processing the audio file: {e}")
        clear_cache()

# Clear memory at the end of processing
clear_cache()

