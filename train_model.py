import os
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Emotion label map
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Path to RAVDESS dataset
dataset_path = "RAVDESS"

# Function to extract MFCC features from a file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

X = []
y = []

print("Extracting features...")

for root, dirs, files in os.walk(dataset_path):
    for file in tqdm(files):
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion = emotion_map.get(emotion_code)
            if emotion:  # Skip unknowns
                features = extract_features(os.path.join(root, file))
                X.append(features)
                y.append(emotion)

print("Training model...")

X = np.array(X)
y = np.array(y)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Test accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

# Save model
joblib.dump(clf, "emotion_model.pkl")
print("Model saved as emotion_model.pkl")
