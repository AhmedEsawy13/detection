import streamlit as st
import pickle
import numpy as np
import os
import librosa
from extract_mfcc import extract_features
import warnings
warnings.filterwarnings("ignore")

model_path = "reciter_models/"
gmm_files = [os.path.join(model_path, reciter_name) for reciter_name in
             os.listdir(model_path) if reciter_name.endswith('.gmm')]

# Load the Gaussian Mixture Models
models = [pickle.load(open(reciter_name, 'r+b')) for reciter_name in gmm_files]
reciters = [reciter_name.split("/")[-1].split(".gmm")[0] for reciter_name in gmm_files]

st.title("Reciter Detection")
st.write("### Welcome to the Reciter Detection ")
uploaded_audio = st.file_uploader("Upload an audio file")
if uploaded_audio:
        st.audio(uploaded_audio)
        signal, sr = librosa.load(uploaded_audio)
        print(signal.shape, sr)
        y = extract_features(signal, sr)
if st.button('Detect'):
        if not uploaded_audio:
            signal, sr = librosa.load("record.wav")
            print(signal.shape, sr)
            y = extract_features(signal, sr)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm = models[i]  # checking with each model one by one
            scores = np.array(gmm.score(y))
            log_likelihood[i] = scores.sum()
        reciter = np.argmax(log_likelihood)
        st.markdown('**' + reciters[reciter] + '**' + " is the reciter detected in this recording.")
