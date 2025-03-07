import streamlit as st
import pandas as pd
import numpy as np
import wfdb
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import MinMaxScaler

# **🔹 Streamlit UI**
st.title("ECG Signal Processing & Feature Extraction")
st.write("Upload an ECG file (`.csv` or `.dat`), adjust parameters, and download results.")

# **🔹 File Upload Option**
uploaded_file = st.file_uploader("Upload your ECG file", type=["csv", "dat"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]  # Get file extension

    if file_type == "csv":
        df_ecg = pd.read_csv(uploaded_file)

        # **Handle Extra Columns**
        if df_ecg.shape[1] > 2:
            df_ecg = df_ecg.iloc[:, [0, 1]]  # Keep only the first two columns
        
        # **Validate CSV Format**
        if df_ecg.shape[1] < 2:
            st.error("Invalid CSV format! Ensure at least two columns: timestamp and raw ECG data.")
            st.stop()
        
        df_ecg.columns = ["Time (ms)", "Raw ECG"]  # Standardizing column names

    elif file_type == "dat":
        record_name = uploaded_file.name.replace(".dat", "")
        record = wfdb.rdrecord(record_name, pn_dir=None)
        raw_signal = record.p_signal[:, 0]
        df_ecg = pd.DataFrame({
            "Time (ms)": np.arange(len(raw_signal)) * (1000 / record.fs),
            "Raw ECG": raw_signal
        })
    else:
        st.error("Unsupported file format!")
        st.stop()

    # **🔹 Hyperparameter Controls in Sidebar**
    st.sidebar.header("🔧 Hyperparameter Tuning")

    # Bandpass Filter Parameters
    filter_order = st.sidebar.slider("Bandpass Filter Order", 1, 10, 4, 1)
    lowcut = st.sidebar.slider("Low Cutoff Frequency (Hz)", 0.5, 5.0, 0.5, 0.1)
    highcut = st.sidebar.slider("High Cutoff Frequency (Hz)", 10, 50, 45, 5)

    # Wavelet Transform Parameters
    wavelet_level = st.sidebar.slider("Wavelet Decomposition Level", 1, 5, 2, 1)
    threshold_factor = st.sidebar.slider("Wavelet Threshold Factor", 0.1, 1.0, 0.4, 0.1)

    # Outlier Removal (IQR)
    iqr_multiplier = st.sidebar.slider("IQR Multiplier (Outlier Removal)", 1.0, 3.0, 1.8, 0.1)

    # R-Peak Detection Parameters
    peak_distance = st.sidebar.slider("Minimum R-Peak Distance (ms)", 100, 500, 600, 10)

    # **🔹 Convert Time to Seconds for Correct Scaling**
    df_ecg["Time (s)"] = df_ecg["Time (ms)"] / 1000  

    # **🔹 Bandpass Filter**
    def butter_bandpass(lowcut, highcut, fs, order):
        nyquist = 0.5 * fs
        low, high = lowcut / nyquist, highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(signal, lowcut, highcut, fs, order):
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        return filtfilt(b, a, signal)

    df_ecg["Raw ECG"] = apply_bandpass_filter(df_ecg["Raw ECG"], lowcut, highcut, 500, filter_order)

    # **🔹 Wavelet Transform for Motion Artifact Removal**
    def remove_motion_artifacts_wavelet(signal):
        coeffs = pywt.wavedec(signal, "db6", level=wavelet_level)
        coeffs[1:] = [pywt.threshold(c, np.std(c) * threshold_factor, mode="soft") for c in coeffs[1:]]
        return pywt.waverec(coeffs, "db6")[:len(signal)]

    df_ecg["Raw ECG"] = remove_motion_artifacts_wavelet(df_ecg["Raw ECG"])

    # **🔹 Outlier Detection using IQR**
    Q1 = df_ecg["Raw ECG"].quantile(0.25)
    Q3 = df_ecg["Raw ECG"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    df_ecg = df_ecg[(df_ecg["Raw ECG"] >= lower_bound) & (df_ecg["Raw ECG"] <= upper_bound)]

    # **🔹 Normalize Before Processing**
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_ecg["ECG Signal"] = scaler.fit_transform(df_ecg[["Raw ECG"]])

    # **🔹 R-Peak Detection**
    fs = 500  
    min_height = np.percentile(df_ecg["ECG Signal"], 90)  # Dynamic threshold
    peak_distance_samples = int(peak_distance / 1000 * fs)  # Convert ms to samples

    peaks, _ = find_peaks(df_ecg["ECG Signal"], distance=peak_distance_samples, height=min_height)
    rr_intervals = np.diff(df_ecg["Time (s)"].iloc[peaks]) * 1000  # Convert to ms

    # **🔹 Extract ECG Features**
    if len(rr_intervals) > 1:
        heart_rate = 60000 / np.mean(rr_intervals)
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        median_rr = np.median(rr_intervals)
    else:
        heart_rate = 0
        mean_rr = 0
        std_rr = 0
        median_rr = 0

    df_features = pd.DataFrame({
        "Heart Rate (BPM)": [heart_rate],
        "Mean RR Interval (ms)": [mean_rr],
        "Median RR Interval (ms)": [median_rr],
        "RR Interval Std Dev (ms)": [std_rr]
    })

    # **🔹 Display Graph**
    st.subheader("ECG Signal with R-Peaks")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_ecg["Time (ms)"], df_ecg["ECG Signal"], label="ECG Signal", color='b')
    ax.scatter(df_ecg["Time (ms)"].iloc[peaks], df_ecg["ECG Signal"].iloc[peaks], color='r', label="R-Peaks", zorder=3)
    ax.legend()
    st.pyplot(fig)

    # **🔹 Download Data**
    st.download_button("Download Processed Data CSV", df_ecg.to_csv(index=False).encode("utf-8"), "processed_ecg.csv", "text/csv")
    st.download_button("Download ECG Features CSV", df_features.to_csv(index=False).encode("utf-8"), "ecg_features.csv", "text/csv")
    st.subheader("Extracted ECG Features")
    st.dataframe(df_features)
