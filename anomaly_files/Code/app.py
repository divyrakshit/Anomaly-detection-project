import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# Load all models and encoders

def load_all_models():
    # LSTM Autoencoder models
    lstm_model = load_model('anomaly_files/LSTM/lstm_autoencoder_anomaly_detection.h5')
    lstm_scaler = joblib.load('anomaly_files/LSTM/standard_scaler.save')

    # CNN Fault Detection models
    cnn_model = load_model('anomaly_files/CNN/cnn_fault_detection.keras')
    cnn_encoder = joblib.load('anomaly_files/CNN/label_encoder_cnn.joblib')

    # ANN Fault Detection models
    ann_model = load_model('anomaly_files/ANN/fault_detection_model.h5')
    ann_encoder = joblib.load('anomaly_files/ANN/label_encoder_aan.joblib')

    return lstm_model, lstm_scaler, cnn_model, cnn_encoder, ann_model, ann_encoder

# Load all models
lstm_model, lstm_scaler, cnn_model, cnn_encoder, ann_model, ann_encoder = load_all_models()

# Signal segmentation function
def segment_signal(signal, win_len, stride=200):
    if len(signal) < win_len:
        return None
    return np.array([signal[i:i + win_len] for i in range(0, len(signal) - win_len + 1, stride)])

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequence = data[i:i+seq_length]
        sequences.append(sequence)
    return np.array(sequences)

# Function to calculate reconstruction error
def calculate_reconstruction_error(model, data):
    predictions = model.predict(data)
    mse = np.mean(np.power(data - predictions, 2), axis=1)
    return mse

# Streamlit app
def main():
    st.title("Anomaly Detection & Fault Classification System")
    st.write("Upload a CSV file containing vibration data for analysis")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.success("File successfully loaded!")

            # Automatically select first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                st.error("No numeric columns found in the CSV file")
                return

            selected_col = numeric_cols[0]
            st.write(f"Analyzing")

            # ========== ANOMALY DETECTION SECTION ==========
            st.header("Step 1: Anomaly Detection")

            # Preprocess the data
            X = df[selected_col].values.reshape(-1, 1)
            X_scaled = lstm_scaler.transform(X)

            # Create sequences
            SEQ_LENGTH = 30
            X_sequences = create_sequences(X_scaled, SEQ_LENGTH)

            if len(X_sequences) == 0:
                st.error(f"Not enough data points to create sequences. Need at least {SEQ_LENGTH} data points.")
                return

            # Calculate reconstruction error
            with st.spinner('Detecting anomalies...'):
                errors = calculate_reconstruction_error(lstm_model, X_sequences)

            # Determine threshold
            threshold = np.percentile(errors, 85)


            # Detect anomalies
            anomalies = errors > threshold

            # Display results
            st.subheader("Anomaly Detection Results")

            # Plot results
            fig, ax = plt.subplots(figsize=(10, 4))
            normal_indices = np.where(~anomalies)[0]
            anomaly_indices = np.where(anomalies)[0]

            ax.plot(normal_indices, errors[normal_indices], 'bo', markersize=3, label='Normal')
            if len(anomaly_indices) > 0:
                ax.plot(anomaly_indices, errors[anomaly_indices], 'ro', markersize=5, label='Anomaly')
            ax.axhline(y=threshold, color='r', linestyle='-', label='Threshold')
            ax.set_title(f'Anomaly Detection Results')
            ax.set_ylabel('Reconstruction Error')
            ax.set_xlabel('Sample Index')
            ax.legend()

            st.pyplot(fig)

            # Summary statistics
            st.write(f"Total samples analyzed: {len(X_sequences)}")
            st.write(f"Number of anomalies detected: {np.sum(anomalies)}")
            st.write(f"Anomaly threshold (85th percentile): {threshold:.4f}")

            # Show anomalies in a table
            if np.sum(anomalies) > 0:
                # Create a dataframe with the original data points marked as anomalies
                result_df = df.copy()
                result_df.columns=['Vibration']
                result_df['Anomaly'] = False
                result_df['Reconstruction_Error'] = np.nan

                # Assign errors to the last point of each sequence
                for idx in range(len(errors)):
                    pos = idx + SEQ_LENGTH - 1
                    if pos < len(result_df):
                        result_df.loc[pos, 'Reconstruction_Error'] = errors[idx]
                        result_df.loc[pos, 'Anomaly'] = anomalies[idx]

                anomaly_df = result_df[result_df['Anomaly'] == True]
                st.subheader("Detected Anomalies")
                st.write(anomaly_df)

                # ========== FAULT CLASSIFICATION SECTION ==========
                st.header("Step 2: Fault Classification")

                if st.button("Classify Detected Anomalies"):
                    with st.spinner('Classifying anomalies...'):
                        # Prepare anomaly segments for classification
                        anomaly_segments = []
                        for idx in anomaly_indices:
                            start_idx = max(0, idx - 500)  # Get 500 points before anomaly
                            end_idx = min(len(X), idx + 500)  # Get 500 points after anomaly
                            segment = X[start_idx:end_idx]
                            anomaly_segments.append(segment)

                        # CNN Classification
                        st.subheader("CNN Model Classification")
                        cnn_predictions = []
                        for seg in anomaly_segments:
                            cnn_windows = segment_signal(seg, win_len=500)
                            if cnn_windows is not None and len(cnn_windows) > 0:
                                preds = cnn_model.predict([cnn_windows]*3)
                                pred_class = np.argmax(preds, axis=1)
                                faults = cnn_encoder.inverse_transform(pred_class)
                                cnn_predictions.extend(faults)

                        if cnn_predictions:
                            fault_types, counts = np.unique(cnn_predictions, return_counts=True)
                            cnn_results = pd.DataFrame({
                                'Fault Type': fault_types,
                                'Count': counts,
                                'Percentage': [f'{c/len(cnn_predictions)*100:.1f}%' for c in counts]
                            })
                            st.table(cnn_results)
                        else:
                            st.warning("CNN couldn't process any anomaly segments")

                        # ANN Classification
                        st.subheader("ANN Model Classification")
                        ann_predictions = []
                        for seg in anomaly_segments:
                            ann_windows = segment_signal(seg, win_len=1000)
                            if ann_windows is not None and len(ann_windows) > 0:
                                preds = ann_model.predict(ann_windows)
                                pred_class = np.argmax(preds, axis=1)
                                faults = ann_encoder.inverse_transform(pred_class)
                                ann_predictions.extend(faults)

                        if ann_predictions:
                            fault_types, counts = np.unique(ann_predictions, return_counts=True)
                            ann_results = pd.DataFrame({
                                'Fault Type': fault_types,
                                'Count': counts,
                                'Percentage': [f'{c/len(ann_predictions)*100:.1f}%' for c in counts]
                            })
                            st.table(ann_results)
                        else:
                            st.warning("ANN couldn't process any anomaly segments")

                # Option to download anomalies
                csv = anomaly_df.to_csv(index=False)
                st.download_button(
                    label="Download Anomalies Report",
                    data=csv,
                    file_name='detected_anomalies.csv',
                    mime='text/csv',
                )
            else:
                st.success("No anomalies detected in the data!")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
