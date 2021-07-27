import os
import pickle
import librosa
import numpy as np
import time
from extract_mfcc import extract_features
import warnings
warnings.filterwarnings("ignore")

source = "test_set/"
model_path = "reciter_models/"
test_file = "Test_Location.txt"
file_paths = open(test_file, 'r')
num_correct_label = 0
num_test_samples = 0
gmm_files = [os.path.join(model_path, reciter_name) for reciter_name in
             os.listdir(model_path) if reciter_name.endswith('.gmm')]

# Load the Gaussian Mixture Models
models = [pickle.load(open(reciter_name, 'r+b')) for reciter_name in gmm_files]
reciters = [reciter_name.split("/")[-1].split(".gmm")[0] for reciter_name in gmm_files]

# Read the test directory and get the list of test audio files
for path in file_paths:
    path = path.strip()
    num_test_samples += 1
    print(path)
    signal, sr = librosa.load(source + path)
    feature_vector = extract_features(signal, sr)
    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i]  # checking with each model one by one
        scores = np.array(gmm.score(feature_vector))
        log_likelihood[i] = scores.sum()
    reciter = np.argmax(log_likelihood)
    if reciters[reciter] == path.split("/")[0]:
        num_correct_label += 1
    print("\tThe Reciter who is detected for this testing sample is ", reciters[reciter])
    time.sleep(1.0)

print('Accuracy:' + str(num_correct_label) + '/' + str(num_test_samples))
