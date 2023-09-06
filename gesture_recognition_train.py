# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 22:38:49 2023

@author: jooer
"""

import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Define the window size and stride
window_size = 100
stride = 50

# Create empty arrays for the features and labels
features  = []
labels  = []

os.chdir(r'C:\Users\jooer\OneDrive\Desktop\IS708_Project_Material\IS708_Project_Material\IS708_API\train_data\train_data\Gesture')
# Loop through each file and extract windows
for gesturer_id in os.listdir():
    for gesture_data_path in os.listdir(os.path.join(os.getcwd(),gesturer_id)):
        df = pd.read_csv(os.path.join(os.getcwd(),gesturer_id,gesture_data_path))
        for i in range(0, len(df) - window_size, stride):
            window = df.iloc[i:i+window_size, 1:]  # Exclude the timestamp column
            features.append(window.values)
            labels.append(gesturer_id)

features = np.array(features)


# Encode labels as integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Convert labels to one-hot encoded format
labels_one_hot = to_categorical(labels_encoded)

# Split data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_one_hot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Define the CNN model architecture
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(window_size, 6)))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

# Compile the model with categorical cross-entropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training set
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)

# Save model
model.save('gesturer_recognition_model.h5')