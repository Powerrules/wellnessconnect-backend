from flask import Flask, request, jsonify
import joblib
import numpy as np
import librosa
import os

app = Flask(__name__)
model = joblib.load("emotion_model.pkl")

@app.route('/api/emotion-detect', methods=['POST'])
def detect_emotion():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    audio_file = request.files['audio']
    audio_path = os.path.join("temp.wav")
    audio_file.save(audio_path)

    # Extract MFCC features
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    prediction = model.predict([mfcc_scaled])[0]

    return jsonify({"emotion": prediction})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
