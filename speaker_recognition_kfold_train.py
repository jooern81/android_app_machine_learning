# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:36:24 2023

@author: jooer
"""

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
import tensorflow as tf

# Load data
data = pd.read_csv('audio_data.csv')
max_length = 16000 * 3 # 3 seconds

# Extract MFCC features and labels
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000, duration=3)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs = librosa.util.fix_length(mfccs, size=max_length)
    return mfccs


features = []
labels = []

for i in range(len(data)):
    file_path = data.iloc[i]['file_path']
    label = data.iloc[i]['speaker_id']
    mfccs = extract_features(file_path)
    features.append(mfccs)
    labels.append(label)

# Encode labels as integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# All data is training data, KFold will split the data
X_train_val, y_train_val = features, labels_encoded

# Create KFold cross-validation splits
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define arrays to store results
test_acc_per_fold = []
test_loss_per_fold = []

# Train and evaluate model using KFold cross-validation
for fold_num, (train_idx, val_idx) in enumerate(kfold.split(X_train_val, y_train_val)):
    # Split data into training and validation sets
    X_train, X_val = np.array(X_train_val)[train_idx], np.array(X_train_val)[val_idx]
    y_train, y_val = np.array(y_train_val)[train_idx], np.array(y_train_val)[val_idx]

    # Build model architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train[0].shape[0], X_train[0].shape[1], 1), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Reshape features to be a 4D array (samples x time steps x features x channels)
    X_train = np.array(X_train).reshape(-1, X_train[0].shape[0], X_train[0].shape[1], 1)
    X_val = np.array(X_val).reshape(-1, X_val[0].shape[0], X_val[0].shape[1], 1)

    # Train model
    print(f"Training fold {fold_num + 1}")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    
    # Evaluate model on test set for this fold
    test_loss, test_acc = model.evaluate(X_val, y_val)
    test_loss_per_fold.append(test_loss)
    test_acc_per_fold.append(test_acc)
    
    
# Print average results over all folds
print('Average test loss:', np.mean(test_loss_per_fold))
print('Average test accuracy:', np.mean(test_acc_per_fold))

# Save model
model.save('speaker_recognition_kfold_model.h5')