import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def extract_features(audio_path):
    """Enhanced feature extraction for early Parkinson's detection"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
       
        features = {'filename': os.path.basename(audio_path)}
       
        # 1. Pitch analysis with early-onset specific ranges
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=60, fmax=500)  # Wider range for early detection
        valid_f0 = f0[~np.isnan(f0)]
       
        if len(valid_f0) > 0:
            features.update({
                'pitch_mean': np.mean(valid_f0),
                'pitch_std': np.std(valid_f0),
                'pitch_range': np.ptp(valid_f0),
                'pitch_variation': np.mean(np.abs(np.diff(valid_f0))),
                'jitter': np.mean(np.abs(np.diff(valid_f0))) / np.mean(valid_f0)  # Relative jitter
            })
        else:
            features.update({k: 0 for k in ['pitch_mean', 'pitch_std', 'pitch_range', 'pitch_variation', 'jitter']})

        # 2. Advanced spectral analysis for early signs
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features.update({
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_flux': np.mean(np.abs(np.diff(spectral_centroids)))  # Important for early detection
        })

        # 3. Speech rhythm analysis (early Parkinson's affects timing)
        rms = librosa.feature.rms(y=y)[0]
        speech_threshold = np.percentile(rms, 75)
        voiced_frames = rms > speech_threshold
        features.update({
            'speech_rate': np.mean(voiced_frames),
            'pause_duration_avg': np.mean(np.diff(np.where(voiced_frames)[0])) if any(voiced_frames) else 0,
            'articulation_rate': len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y)/sr)  # Syllables per second
        })

        # 4. Voice quality metrics
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        features.update({
            'hnr': np.mean(harmonic) / (np.mean(percussive) + 1e-6),
            'shimmer': np.mean(np.abs(np.diff(rms[voiced_frames]))) if any(voiced_frames) else 0,
            'voice_breaks': len(librosa.onset.onset_detect(y=harmonic, sr=sr))  # Voice breaks count
        })

        # 5. Comprehensive MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # More coefficients for early detection
        for i in range(5):  # First 5 MFCCs are most important
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc{i+1}_std'] = np.std(mfcc[i])

        return features
   
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def analyze_dataset(audio_dir, output_file="parkinsons_results.csv"):
    """Analyze all audio files in directory for early Parkinson's signs"""
    audio_files = [f for f in os.listdir(audio_dir) 
                  if f.lower().endswith(('.mp3', '.wav', '.ogg', '.flac'))]
    
    if not audio_files:
        print("No audio files found in directory")
        return
    
    print(f"Found {len(audio_files)} audio files. Processing...")
    
    # Extract features with progress bar
    all_features = []
    for file in tqdm(audio_files, desc="Processing audio files"):
        features = extract_features(os.path.join(audio_dir, file))
        if features:
            all_features.append(features)
    
    if not all_features:
        print("No features could be extracted from any files")
        return
    
    # Create DataFrame and assess Parkinson's risk
    df = pd.DataFrame(all_features)
    df = assess_parkinsons_risk(df)
    
    # Save results
    output_path = os.path.join(audio_dir, output_file)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Generate visualizations
    generate_analysis_report(df, audio_dir)
    
    return df

def assess_parkinsons_risk(features_df):
    """Assess early-onset Parkinson's risk using clinically-informed thresholds"""
    # Early-onset specific thresholds (based on research)
    thresholds = {
        'pitch_std': {'threshold': 12, 'direction': 'lower', 'weight': 0.15},
        'pitch_variation': {'threshold': 1.5, 'direction': 'lower', 'weight': 0.15},
        'jitter': {'threshold': 0.008, 'direction': 'higher', 'weight': 0.15},
        'spectral_flux': {'threshold': 50, 'direction': 'higher', 'weight': 0.1},
        'hnr': {'threshold': 0.05, 'direction': 'lower', 'weight': 0.1},
        'shimmer': {'threshold': 0.15, 'direction': 'higher', 'weight': 0.1},
        'articulation_rate': {'threshold': 4.5, 'direction': 'lower', 'weight': 0.1},
        'mfcc1_std': {'threshold': 10, 'direction': 'higher', 'weight': 0.1},
        'voice_breaks': {'threshold': 5, 'direction': 'higher', 'weight': 0.05}
    }
    
    results = []
    for _, row in features_df.iterrows():
        risk_score = 0
        markers = []
        
        for feature, params in thresholds.items():
            if feature not in row:
                continue
                
            value = row[feature]
            if params['direction'] == 'lower' and value < params['threshold']:
                risk_score += params['weight']
                markers.append(f"Low {feature.replace('_', ' ')}")
            elif params['direction'] == 'higher' and value > params['threshold']:
                risk_score += params['weight']
                markers.append(f"High {feature.replace('_', ' ')}")
        
        # Early-onset specific classification
        if risk_score > 0.65:
            status = "High risk of early-onset Parkinson's"
        elif risk_score > 0.4:
            status = "Moderate risk of early Parkinson's"
        elif risk_score > 0.2:
            status = "Possible early signs"
        else:
            status = "Normal"
        
        results.append({
            'risk_score': round(risk_score, 3),
            'risk_status': status,
            'key_markers': ", ".join(markers[:3]) if markers else "No significant markers",
            **row
        })
    
    return pd.DataFrame(results)

def generate_analysis_report(df, output_dir):
    """Generate comprehensive visual report"""
    plt.figure(figsize=(18, 12))
    
    # 1. Risk distribution
    plt.subplot(2, 2, 1)
    df['risk_status'].value_counts().plot(kind='bar', color=['green', 'yellow', 'orange', 'red'])
    plt.title('Early Parkinson\'s Risk Distribution')
    plt.ylabel('Number of samples')
    
    # 2. Feature importance
    plt.subplot(2, 2, 2)
    features = ['jitter', 'pitch_variation', 'spectral_flux', 'hnr', 'articulation_rate']
    avg_values = df[features].mean()
    avg_values.plot(kind='barh')
    plt.title('Average Values of Key Early Markers')
    
    # 3. Risk score distribution
    plt.subplot(2, 2, 3)
    plt.hist(df['risk_score'], bins=20)
    plt.title('Distribution of Risk Scores')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    
    # 4. Feature correlation
    plt.subplot(2, 2, 4)
    plt.scatter(df['pitch_variation'], df['jitter'], c=df['risk_score'], cmap='viridis')
    plt.colorbar(label='Risk Score')
    plt.xlabel('Pitch Variation')
    plt.ylabel('Jitter')
    plt.title('Key Feature Relationship')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'early_parkinsons_analysis.png'))
    plt.show()

if __name__ == "__main__":
    audio_directory = r"E:\AIMD_Parkinsons"
    
    if not os.path.exists(audio_directory):
        print(f"Directory not found: {audio_directory}")
    else:
        print(f"Analyzing audio files for early Parkinson's detection in: {audio_directory}")
        results = analyze_dataset(audio_directory)
        
        if results is not None:
            print("\n=== Sample Risk Assessment ===")
            print(results[['filename', 'risk_score', 'risk_status', 'key_markers']].head())