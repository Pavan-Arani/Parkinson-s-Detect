import librosa
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ThreadPoolExecutor


def extract_features(audio_path):
    """Extract relevant features for Parkinson's detection from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=5)  # Limit to first 5 seconds for speed
        features = {}
        
        # Pitch-related features
        f0 = librosa.yin(y, fmin=75, fmax=600)
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) > 0:
            features['pitch_mean'] = np.mean(valid_f0)
            features['pitch_std'] = np.std(valid_f0)
            features['pitch_range'] = np.max(valid_f0) - np.min(valid_f0)
        else:
            features['pitch_mean'] = features['pitch_std'] = features['pitch_range'] = 0
        
        # Jitter (voice instability measure)
        features['jitter'] = np.mean(np.abs(np.diff(valid_f0))) if len(valid_f0) > 1 else 0
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Speech rate
        rms = librosa.feature.rms(y=y)[0]
        speech_threshold = np.percentile(rms, 70)
        speech_frames = np.sum(rms > speech_threshold)
        features['speech_rate'] = speech_frames / len(rms)
        
        # MFCC features (first coefficient)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)  # Reduce MFCC count for speed
        features['mfcc1_mean'] = np.mean(mfcc[0])
        
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None


def process_directory_with_threads(audio_dir, n_threads=4):
    """Process files using multithreading to reduce CPU usage."""
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.mp3', '.wav', '.ogg'))]
    file_paths = [os.path.join(audio_dir, f) for f in audio_files]
    
    print(f"Processing {len(audio_files)} audio files...")

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(extract_features, file_paths))
    
    data = []
    for idx, (file, features) in enumerate(zip(audio_files, results)):
        print(f"Processed file {idx+1}/{len(audio_files)}: {file}")  # Track progress
        if features:
            features['file_name'] = file  # Add filename for reference
            data.append(features)
    
    print("Feature extraction complete.")
    return pd.DataFrame(data)


def train_model(data, labels):
    """Train a machine learning model on extracted features."""
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2)  # Limit parallel jobs
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    return model


if __name__ == "__main__":
    audio_directory = "C:\\Users\\vikas\\Downloads\\cv-corpus-19.0-delta-2024-09-13\\en\\clips"
    if not os.path.exists(audio_directory):
        print(f"Directory not found: {audio_directory}")
    else:
        print(f"Analyzing audio files in: {audio_directory}")
        df = process_directory_with_threads(audio_directory)
        
        # Assuming we have labels for training (1 = Parkinson's, 0 = Healthy)
        labels = [0] * len(df)  # Placeholder: Replace with real labels
        
        if len(df) > 0:
            model = train_model(df.drop(columns=['file_name']), labels)
        else:
            print("No valid audio features extracted.")
