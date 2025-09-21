import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and label encoder
model = load_model('/content/drive/MyDrive/anomaly_files/ANN/fault_detection_model.h5')
encoder = joblib.load('/content/drive/MyDrive/anomaly_files/ANN/label_encoder.joblib')

# Function to segment the signal into windows
def segment_signal(signal, win_len=1000, stride=200):
    if len(signal) < win_len:
        return None  # Signal too short
    windows = []
    for i in range(0, len(signal) - win_len + 1, stride):
        window = signal[i:i + win_len]
        windows.append(window)
    return np.array(windows)

# Streamlit app
st.title("Fault Detection Prediction App")
st.write("Upload a CSV file with a single column of vibration signal data (at least 1000 data points).")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV
    try:
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape[1] != 1:
            st.error("Error: CSV must contain exactly one column of vibration data.")
        else:
            signal = df.iloc[:, 0].values  # Extract the single column as a NumPy array
            st.write(f"Loaded signal with {len(signal)} data points.")

            # Preprocess the signal
            windows = segment_signal(signal)
            if windows is None:
                st.error("Error: Signal is too short. It must have at least 1000 data points.")
            else:
                st.write(f"Segmented into {windows.shape[0]} windows of size 1000.")

                # Make predictions
                predictions = model.predict(windows)  # Shape: (num_windows, num_classes)
                predicted_classes = np.argmax(predictions, axis=1)  # Most likely class per window
                predicted_faults = encoder.inverse_transform(predicted_classes)  # Decode to fault names

                # Calculate distribution
                unique_faults, counts = np.unique(predicted_faults, return_counts=True)
                most_frequent_fault = unique_faults[np.argmax(counts)]
                distribution = {fault: count / len(predicted_faults) * 100 for fault, count in zip(unique_faults, counts)}

                # Display results
                st.subheader("Prediction Results")
                st.write(f"**Predicted Fault Type**: {most_frequent_fault}")
                st.write("**Prediction Distribution Across Windows**:")
                for fault, percentage in distribution.items():
                    st.write(f"{fault}: {percentage:.2f}%")

                # Confidence (average max probability for the most frequent fault)
                confidences = np.max(predictions, axis=1)  # Max probability per window
                mask = predicted_faults == most_frequent_fault
                avg_confidence = np.mean(confidences[mask]) * 100 if mask.any() else 0
                st.write(f"**Average Confidence for {most_frequent_fault}**: {avg_confidence:.2f}%")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a CSV file to proceed.")