from flask import Flask, request, render_template, redirect, url_for
import joblib
import librosa
import numpy as np
import os

app = Flask(__name__)

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return np.array([mfccs])
    except Exception as e:
        print("Error encountered while processing file:", file_path)
        return None

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        audio_file = request.files['file']
        if not os.path.exists('choosenaudios'):
            os.makedirs('choosenaudios')
        audio_file.save(os.path.join('choosenaudios', audio_file.filename))
        loaded_model = joblib.load("random_forest_modelAudioMixed2NewMoredata2.joblib")
        example_features = extract_features(os.path.join('choosenaudios', audio_file.filename))
        if example_features is not None:
            proba = loaded_model.predict_proba(example_features)
            proba_fake, proba_real = proba[0]
            max_proba_class = "Real" if proba_real > proba_fake else "Fake"
            result = f"{max_proba_class} Audio File"
        else:
            result = "Error extracting features from the audio file."
        return render_template('predict.html', prediction_text=result)
    return render_template('predict.html', prediction_text='Upload an audio file to predict.')

if __name__ == '__main__':
    app.run(debug=True)