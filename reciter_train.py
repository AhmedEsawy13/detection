import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from extract_mfcc import extract_features
import warnings
import librosa
warnings.filterwarnings("ignore")

source = "train_set/"
dest = "reciter_models/"
train_file = "Train_Location.txt"
file_paths = open(train_file, 'r')

count = 1
# Extracting features for each reciter (387 files per reciters)
features = np.asarray(())
for path in file_paths:
    path = path.strip()
    print(path)

    # read the audio
    signal, sr = librosa.load(source + path)
    # extract 40 dimensional MFCC & delta MFCC features
    vector = extract_features(signal, sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 387 files of reciter are concatenated, then do model training
    if count == 387:
        gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
        gmm.fit(features)
        # dumping the trained gaussian model
        pickle_file = path.split("/")[0] + ".gmm"
        pickle.dump(gmm, open(dest + pickle_file, 'wb'))
        print('+ complete modeling for reciter:', pickle_file, " with data point = ", features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1
