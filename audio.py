import os
import numpy as np
import librosa
import pandas as pd
from pydub import AudioSegment
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import parselmouth

dataset_dir = "C:\\path\\to\\dataset"

def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")
    if not os.path.exists(wav_path):  
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
    return wav_path


def extract_features(audio_file):
    audio_path = os.path.join(dataset_dir, audio_file)

    if audio_file.endswith(".mp3"):
        audio_path = convert_mp3_to_wav(audio_path)

    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=5)  
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)

        sound = parselmouth.Sound(audio_path)
        pitch = sound.to_pitch()
        f0_mean = np.mean(pitch.selected_array['frequency']) if pitch else 0
        jitter = sound.to_jitter() if sound.to_jitter() else 0
        shimmer = sound.to_shimmer() if sound.to_shimmer() else 0
        hnr = sound.to_harmonicity() if sound.to_harmonicity() else 0

        label = 1 if "parkinsons" in audio_file.lower() else 0  
        return [audio_file] + list(mfccs) + [f0_mean, jitter, shimmer, hnr, label]

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None


audio_files = [f for f in os.listdir(dataset_dir) if f.endswith((".mp3", ".wav"))]


with Pool(cpu_count() - 1) as pool:
    data = pool.map(extract_features, audio_files)


data = [d for d in data if d is not None]


columns = ["filename"] + [f"mfcc_{i}" for i in range(13)] + ["f0_mean", "jitter", "shimmer", "hnr", "label"]
df = pd.DataFrame(data, columns=columns)


df.to_parquet("parkinsons_audio_features.parquet", index=False)


df = pd.read_parquet("parkinsons_audio_features.parquet")

X = df.drop(columns=["filename", "label"])
y = df["label"]

pca = PCA(n_components=10) 
X_reduced = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

clf = LGBMClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

new_sample = np.array(X_test[0]).reshape(1, -1)
prediction = clf.predict(new_sample)
print(f"Prediction for sample: {'Parkinson' if prediction[0] == 1 else 'Healthy'}")
