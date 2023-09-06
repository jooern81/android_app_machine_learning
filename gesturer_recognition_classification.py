# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 23:06:56 2023

@author: jooer
"""
import os
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the saved model
model = load_model('gesturer_recognition_model.h5')


# Load the label_encoder object
with open('gesturer_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
    
# Preprocess the input data to match the training data format
window_size = 100
stride = 50

# Takes in a dataframe of time_index, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z and returns dict of gesturer confidence
def extract_windows_and_predict(input_data):
    input_data.fillna(0.00, inplace=True)
    input_data = input_data[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    input_data_windows = []
    for i in range(0, len(input_data) - window_size, stride):
        window = input_data.iloc[i:i+window_size]
        input_data_windows.append(window.values)

    predictions = model.predict(np.array(input_data_windows))
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_labels = list(label_encoder.inverse_transform(predicted_labels))
    label_confidence = {1:predicted_labels.count('gesturer_1')/len(predicted_labels),2:predicted_labels.count('gesturer_2')/len(predicted_labels),3:predicted_labels.count('gesturer_3')/len(predicted_labels),4:predicted_labels.count('gesturer_4')/len(predicted_labels),5:predicted_labels.count('gesturer_5')/len(predicted_labels)}
    
    return label_confidence



# Function test
# df = pd.read_csv(r'C:\Users\jooer\OneDrive\Desktop\IS708_Project_Material\IS708_Project_Material\IS708_API\gesture_data_1680539147107938.csv')
# df.fillna(0, inplace=True)
# print(df)
# x = extract_windows_and_predict(pd.read_csv(r'C:\Users\jooer\OneDrive\Desktop\IS708_Project_Material\IS708_Project_Material\IS708_API\gesture_data_1680539147107938.csv'))
# print(x)
# x = extract_windows_and_predict(pd.read_csv(r'C:\Users\jooer\OneDrive\Desktop\IS708_Project_Material\IS708_Project_Material\IS708_API\train_data\train_data\Gesture\5\1678081738580.csv'))
# print(x)

# correct = 0
# # Full test run on dataset
# test_data_path = r'C:\Users\jooer\OneDrive\Desktop\IS708_Project_Material\IS708_Project_Material\IS708_API\train_data\train_data\Gesture'
# for gesturer_id in os.listdir(test_data_path):
#     for gesture_data_path in os.listdir(os.path.join(test_data_path,gesturer_id)):
#         df = pd.read_csv(os.path.join(test_data_path,gesturer_id,gesture_data_path))
#         input_data_windows = []
#         for i in range(0, len(df) - window_size, stride):
#             window = df.iloc[i:i+window_size, 1:]  # Exclude the timestamp column
#             input_data_windows.append(window.values)
#         predictions = model.predict(np.array(input_data_windows))
#         predicted_labels = np.argmax(predictions, axis=1)
#         predicted_labels = list(label_encoder.inverse_transform(predicted_labels))
#         label_confidence = {1:predicted_labels.count('gesturer_1')/len(predicted_labels),2:predicted_labels.count('gesturer_2')/len(predicted_labels),3:predicted_labels.count('gesturer_3')/len(predicted_labels),4:predicted_labels.count('gesturer_4')/len(predicted_labels),5:predicted_labels.count('gesturer_5')/len(predicted_labels)}
        
#         print('Confidence Values:', label_confidence)
#         print(max(label_confidence, key=label_confidence.get), gesturer_id)
#         if max(label_confidence, key=label_confidence.get) == int(gesturer_id): correct += 1
        
# print(correct/500)

