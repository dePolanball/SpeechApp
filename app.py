
import streamlit as st
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import requests
import librosa
import numpy as np

# Load models for transcription and phoneme analysis
@st.cache_resource
def load_phoneme_model():
    # Load Wav2Vec2-Phoneme model and processor from Huggingface
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    return processor, model

@st.cache_resource
def load_asr_model():
    # Load Huggingface Whisper model for speech-to-text transcription
    asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-base")
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

# Load the models
processor, phoneme_model = load_phoneme_model()
asr_model = load_asr_model()

# Streamlit app interface
st.title("Speech Transcription, Phoneme Analysis, and Grammar Evaluation")

st.write("Upload an audio file (WAV, MP3, OGG) and get a transcription, pronunciation evaluation via phoneme analysis, and grammar check.")

# File uploader for audio files
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Play the uploaded audio file
    st.audio(uploaded_file, format=uploaded_file.type)

    # Load the audio file using librosa
    y, sr = librosa.load(uploaded_file, sr=16000)

    # Transcribe the audio using Huggingface Whisper model
    with st.spinner("Transcribing..."):
        # Pass the correct dictionary format to the ASR model
        inputs = {"raw": y, "sampling_rate": sr}
        transcription = asr_model(inputs)['text']
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

