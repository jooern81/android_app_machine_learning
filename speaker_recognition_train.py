import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
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
    
# normalize the 3D array using StandardScaler
#scaler = StandardScaler()
#features = np.array(features)
#features = scaler.fit_transform(features.reshape(features.shape[0], -1)).reshape(features.shape)
    
# Encode labels as integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)


# Split data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


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
model.add(layers.Dense(5, activation='softmax'))


# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape features to be a 4D array (samples x time steps x features x channels)
X_train = np.array(X_train).reshape(-1, X_train[0].shape[0], X_train[0].shape[1], 1)
X_val = np.array(X_val).reshape(-1, X_val[0].shape[0], X_val[0].shape[1], 1)
X_test = np.array(X_test).reshape(-1, X_test[0].shape[0], X_test[0].shape[1], 1)

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate model on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Save model
model.save('speaker_recognition_model.h5')