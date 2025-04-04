import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def extract_features(audio_path):
    """Extract relevant features for Parkinson's detection from audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
       
        features = {}
       
        # 1. Pitch-related features
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=600)
        valid_f0 = f0[~np.isnan(f0)]
       
        if len(valid_f0) > 0:
            features['pitch_mean'] = np.mean(valid_f0)
            features['pitch_std'] = np.std(valid_f0)
            features['pitch_range'] = np.max(valid_f0) - np.min(valid_f0)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
       
        # 2. Jitter (voice instability measure)
        jitter = np.mean(np.abs(np.diff(valid_f0))) if len(valid_f0) > 1 else 0
        features['jitter'] = jitter
       
        # 3. Formant frequencies (simplified version)
        # We'll use spectral peaks instead of the problematic peak_pick
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
       
        # 4. Speech rate and pause analysis
        rms = librosa.feature.rms(y=y)[0]
        speech_threshold = np.percentile(rms, 70)
        speech_frames = np.sum(rms > speech_threshold)
        features['speech_rate'] = speech_frames / len(rms)
       
        # 5. Harmonics-to-noise ratio (HNR)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        features['hnr'] = np.mean(harmonic) / (np.mean(percussive) + 1e-6)
       
        # 6. MFCC features (first coefficient)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc1_mean'] = np.mean(mfcc[0])
       
        return features
   
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None


def compare_audios(audio_dir):
    """Compare two audio files in the specified directory"""
    # Get all audio files in directory
    audio_files = [f for f in os.listdir(audio_dir)
                  if f.lower().endswith(('.mp3', '.wav', '.ogg'))]
   
    if len(audio_files) < 2:
        print("Need at least 2 audio files for comparison")
        return
   
    # Limit to first two files
    audio_files = audio_files[:2]
    file_paths = [os.path.join(audio_dir, f) for f in audio_files]
   
    # Extract features for both files
    features = []
    for path in file_paths:
        print(f"\nProcessing: {os.path.basename(path)}")
        feat = extract_features(path)
        if feat:
            features.append(feat)
            print("Features extracted successfully")
        else:
            print("Failed to extract features")
   
    if len(features) != 2:
        print("Couldn't extract features from both files")
        return
   
    # Create comparison table
    print("\n\n=== Audio Feature Comparison ===")
    print("{:<25} {:<15} {:<15} {:<10}".format(
        "Feature", "File 1", "File 2", "Difference"))
    print("-" * 70)
   
    parkinson_markers = {
        'pitch_std': "Lower values may indicate Parkinson's",
        'pitch_range': "Reduced range may indicate Parkinson's",
        'jitter': "Higher values may indicate Parkinson's",
        'speech_rate': "Lower values may indicate Parkinson's",
        'hnr': "Lower values may indicate Parkinson's",
        'spectral_centroid_std': "Higher values may indicate tremor"
    }
   
    for key in features[0].keys():
        val1 = features[0][key]
        val2 = features[1][key]
        diff = abs(val1 - val2)
       
        note = parkinson_markers.get(key, "")
       
        print("{:<25} {:<15.2f} {:<15.2f} {:<10.2f} {:<30}".format(
            key, val1, val2, diff, note))
   
    # Generate simple visualization
    plot_features(features, [os.path.basename(f) for f in file_paths])


def plot_features(features, labels):
    """Visualize the comparison of key features"""
    keys = ['pitch_std', 'pitch_range', 'jitter', 'speech_rate', 'hnr', 'spectral_centroid_std']
   
    plt.figure(figsize=(14, 8))
    for i, key in enumerate(keys):
        plt.subplot(2, 3, i+1)
        values = [f.get(key, 0) for f in features]
        plt.bar(labels, values)
        plt.title(key.replace('_', ' ').title())
        plt.xticks(rotation=45)
   
    plt.tight_layout()
    plt.suptitle("Parkinson's Disease Voice Marker Comparison", y=1.02)
    plt.show()


if __name__ == "__main__":
    audio_directory = r"E:\AIMD_Parkinsons"
   
    # Verify directory exists
    if not os.path.exists(audio_directory):
        print(f"Directory not found: {audio_directory}")
    else:
        print(f"Analyzing audio files in: {audio_directory}")
        compare_audios(audio_directory)
