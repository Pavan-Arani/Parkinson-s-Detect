import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import tempfile
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Parkinson's Voice Detection",
    page_icon="🎤",
    layout="wide"
)

# Load the saved model package
@st.cache_resource
def load_model():
    model_package = joblib.load("parkinson_models.pkl")
    return model_package

model_package = load_model()

# Define the feature extraction function (same as in your notebook)
def extract_features(audio_path):
    """Extract features ensuring all are scalar values"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        features = {}

        # 1. Pitch features
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=600)
        valid_f0 = f0[~np.isnan(f0)]
        
        # Pitch statistics (ensure scalar outputs)
        features['pitch_mean'] = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
        features['pitch_std'] = float(np.std(valid_f0)) if len(valid_f0) > 0 else 0.0
        features['pitch_range'] = float(np.ptp(valid_f0)) if len(valid_f0) > 0 else 0.0

        # 2. Jitter and shimmer (scalar)
        if len(valid_f0) > 1:
            features['jitter'] = float(np.mean(np.abs(np.diff(valid_f0))))
            amplitude = np.abs(librosa.stft(y))
            features['shimmer'] = float(np.mean(np.abs(np.diff(amplitude, axis=1))))
        else:
            features['jitter'] = 0.0
            features['shimmer'] = 0.0

        # 3. Voice quality measures (scalar)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        features['hnr'] = float(np.mean(harmonic) / (np.mean(percussive) + 1e-6))
        features['tremor_index'] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))

        # 4. Spectral features (scalar)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))

        # 5. MFCCs (convert to individual features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
        for i in range(5):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))

        # 6. Rhythm features (scalar)
        rms = librosa.feature.rms(y=y)[0]
        speech_threshold = np.percentile(rms, 70)
        features['speech_rate'] = float(np.sum(rms > speech_threshold) / len(rms))
        
        return features
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

# Streamlit UI
def main():
    st.title("🎤 Parkinson's Disease Detection from Voice")
    st.markdown("""
    This app analyzes voice recordings to detect potential signs of Parkinson's disease.
    Upload an audio file (.wav, .mp3) to get a prediction.
    """)
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        list(model_package['models'].keys()),
        index=0
    )
    
    selected_model = model_package['models'][model_name]['model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=['wav', 'mp3', 'ogg']
    )
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Display audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Extract features
            with st.spinner("Analyzing audio features..."):
                features = extract_features(tmp_path)
                
                if features:
                    # Create a DataFrame with the same structure as training data
                    features_df = pd.DataFrame([features])
                    
                    # Prepare features (drop filename and label if present)
                    features_df = features_df.drop(['filename', 'label'], axis=1, errors='ignore')
                    
                    # Ensure all expected features are present
                    for feature in feature_names:
                        if feature not in features_df.columns:
                            features_df[feature] = 0.0  # Add missing features with default value
                    
                    # Reorder columns to match training data
                    features_df = features_df[feature_names]
                    
                    # Scale features
                    features_scaled = scaler.transform(features_df)
                    
                    # Make prediction
                    prediction = selected_model.predict(features_scaled)
                    proba = selected_model.predict_proba(features_scaled)
                    
                    # Display results
                    st.subheader("Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Prediction", 
                                 model_package['class_names'][prediction[0]],
                                 f"{proba[0][prediction[0]]*100:.1f}% confidence")
                        
                        # Show probability distribution
                        fig, ax = plt.subplots()
                        ax.bar(model_package['class_names'], proba[0])
                        ax.set_ylabel("Probability")
                        ax.set_title("Prediction Probabilities")
                        st.pyplot(fig)
                    
                    with col2:
                        # Show top features
                        if hasattr(selected_model, 'feature_importances_'):
                            importances = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': selected_model.feature_importances_
                            }).sort_values('Importance', ascending=False).head(10)
                            
                            st.write("Top influential features:")
                            st.dataframe(importances)
                        
                        # Show raw feature values
                        if st.checkbox("Show extracted features"):
                            st.dataframe(features_df.T.rename(columns={0: 'Value'}))
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

if __name__ == "__main__":
    main()

