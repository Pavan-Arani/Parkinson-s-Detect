# Parkinsons Detect

2nd Place team during AIMD Spring 2025 Presentation Night, UTD

Credit - Romita Veeramallu, Shariq Hasan, Rajat , Pavan Arani, Krithikaa Muthukumar, Vikas Mariyappa


## What's the Goal

A low-cost, and accessible tool to support pre-clinical screening. This voice-based machine learning tool built to assist in the early detection of Parkinson’s Disease by analyzing subtle vocal biomarkers.
Designed as a semester-long project by a team of 6, the system decodes subtle vocal disturbances spotted in early diagnosis.

## How does it work

- Upload a .wav file, no special hardware required

- Let the model extract biomarkers and generate a Parkinson’s likelihood score

- See how each vocal feature contributes to the output

- Access everything through a clean Streamlit interface

Voice is a complex signal filled with patterns that change when something’s off in the brain. ParkinsonDetect captures those tiny disruptions using:

Pitch, Speech rate, Jitter (frequency irregularities), Shimmer (amplitude variations), Harmonics-to-noise ratio, and Tremor frequencies

We ran these through two models for performance and interpretability to create a machine learning model:

Random Forest and Support Vector Machine (SVM), Both trained on curated clinical datasets for Parkinson’s-related speech changes.

Our Final accuracy was greater than or equal 85% on average

## Technologies Used

Librosa - A powerful audio analysis library that enabled extraction of nuanced vocal features like jitter, shimmer, and frequency irregularities.

Scikit-learn - Used to build, train, and evaluate classification models like Random Forest and SVM with extensive support for tuning and validation.

NumPy and Pandas - Provided efficient data structures and operations for organizing, transforming, and managing the extracted audio features.

Streamlit - Allowed for rapid development of an interactive UI, and for a browser-based diagnostic tool for real-time audio upload and prediction.

Matplotlib and Seaborn - Enabled us to create clear, informative visualizations of feature distributions and model outputs to aid interpretability.

