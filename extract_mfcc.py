import numpy as np
from sklearn import preprocessing
import librosa


def extract_features(signal, sr):
    mfcc_feature = librosa.feature.mfcc(signal, n_mfcc=20, sr=sr)
    mfcc_feature = preprocessing.scale(mfcc_feature, axis=1)
    mfcc_feature = np.transpose(mfcc_feature)
    delta = librosa.feature.delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined
