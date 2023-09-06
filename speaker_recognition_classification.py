# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:22:07 2023

@author: jooer
"""

import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import pandas as pd


# Extracts feature from .wav audio file and returns dict containing label confidence values
def extract_features_and_predict(file_path):

    # Load saved model
    model = tf.keras.models.load_model('speaker_recognition_model.h5')
    
    # Load the label_encoder object
    with open('audio_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
        
    # Extract MFCC features and labels
    max_length = 16000 * 3 # 3 seconds  
        
    audio, sr = librosa.load(file_path, sr=16000, duration=3)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs = librosa.util.fix_length(mfccs, size=max_length)


    # Reshape and scale MFCC features
    mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
    #mfccs = scaler.fit_transform(mfccs.reshape(-1, 1)).reshape(mfccs.shape)
    
    # Predict class of input file
    prediction = model.predict(mfccs)
    print(prediction)
    
    label_confidence = {}
    for index,class_confidence in enumerate(prediction[0]):
        label_confidence[label_encoder.inverse_transform([index])[0]] = class_confidence
    
    return label_confidence


# Test run model on full data
# data = pd.read_csv('audio_data.csv')
# correct_predictions = 0
# for file_path,speaker_id in zip(data.file_path,data.speaker_id):
#     prediction = extract_features_and_predict(file_path)
#     print(speaker_id,prediction)
#     if speaker_id == prediction: correct_predictions += 1

# print('Prediction Accuracy:',correct_predictions/len(data.file_path))
    
    