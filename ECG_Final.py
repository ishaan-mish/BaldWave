import streamlit as st
import serial
import csv
import time
import pandas as pd
import numpy as np
import wfdb
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import MinMaxScaler
import threading

# 📌 Serial Port Configuration
SERIAL_PORT = "COM2"  # Change this based on your system
BAUD_RATE = 115200
OUTPUT_FILE = "ecg_data.csv"

# 📌 Streamlit App
st.title("Real-Time ECG Data Collection & Processing")

# ✅ **Step 1: Start/Stop Data Collection**
collecting = st.session_state.get("collecting", False)

def collect_ecg_data():
    """ Function to collect ECG data from Arduino & save to CSV. """
    global collecting
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Allow time for connection
        
        with open(OUTPUT_FILE, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time (ms)", "Raw ECG"])  # Write header
            
            while collecting:
                line = ser.readline().decode("utf-8").strip()
                if line and "," in line:
                    writer.writerow(line.split(","))  # Save data
        ser.close()
    except Exception as e:
        st.error(f"Error: {e}")

if st.button("Start ECG Data Collection", disabled=collecting):
    collecting = True
    st.session_state["collecting"] = True
    thread = threading.Thread(target=collect_ecg_data, daemon=True)
    thread.start()
    st.success("ECG Data Collection Started...")

if st.button("Stop & Save Data", disabled=not collecting):
    collecting = False
    st.session_state["collecting"] = False
    st.success(f"ECG Data Saved as `{OUTPUT_FILE}`!")

# ✅ **Step 2: Display Raw Data Before Processing**
if not collecting and st.session_state.get("collecting") is False and OUTPUT_FILE:
    st.subheader("📊 Raw ECG Data Visualization")

    # **Read ECG CSV**
    df_ecg = pd.read_csv(OUTPUT_FILE)

    # **Validate CSV**
    if df_ecg.shape[1] < 2:
        st.error("Invalid CSV format! Ensure at least two columns: timestamp and raw ECG data.")
        st.stop()

    df_ecg.columns = ["Time (ms)", "Raw ECG"]

    # ✅ **Plot Raw ECG Signal**
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_ecg["Time (ms)"], df_ecg["Raw ECG"], label="Raw ECG Signal", color='b')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Raw ECG Value")
    ax.set_title("Raw ECG Signal from Arduino")
    ax.legend()
    st.pyplot(fig)

    # ✅ **Proceed with Processing**
    st.subheader("⚙️ Processing ECG Data...")

    # ✅ **Hyperparameter Tuning UI**
    st.sidebar.header("🔧 Hyperparameter Tuning")

    filter_order = st.sidebar.slider("Bandpass Filter Order", 1, 10, 4, 1)
    lowcut = st.sidebar.slider("Low Cutoff Frequency (Hz)", 0.5, 5.0, 0.5, 0.1)
    highcut = st.sidebar.slider("High Cutoff Frequency (Hz)", 10, 50, 45, 5)
    wavelet_level = st.sidebar.slider("Wavelet Decomposition Level", 1, 5, 2, 1)
    threshold_factor = st.sidebar.slider("Wavelet Threshold Factor", 0.1, 1.0, 0.4, 0.1)
    iqr_multiplier = st.sidebar.slider("IQR Multiplier (Outlier Removal)", 1.0, 3.0, 1.8, 0.1)
    peak_distance = st.sidebar.slider("Minimum R-Peak Distance (ms)", 100, 500, 600, 10)

    # ✅ **Convert Time to Seconds**
    df_ecg["Time (s)"] = df_ecg["Time (ms)"] / 1000  

    # ✅ **Bandpass Filter**
    def butter_bandpass(lowcut, highcut, fs, order):
        nyquist = 0.5 * fs
        low, high = lowcut / nyquist, highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(signal, lowcut, highcut, fs, order):
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        return filtfilt(b, a, signal)

    df_ecg["Filtered ECG"] = apply_bandpass_filter(df_ecg["Raw ECG"], lowcut, highcut, 500, filter_order)

    # ✅ **Wavelet Transform for Motion Artifact Removal**
    def remove_motion_artifacts_wavelet(signal):
        coeffs = pywt.wavedec(signal, "db6", level=wavelet_level)
        coeffs[1:] = [pywt.threshold(c, np.std(c) * threshold_factor, mode="soft") for c in coeffs[1:]]
        return pywt.waverec(coeffs, "db6")[:len(signal)]

    df_ecg["Processed ECG"] = remove_motion_artifacts_wavelet(df_ecg["Filtered ECG"])

    # ✅ **Plot Processed ECG Signal**
    st.subheader("📊 Processed ECG Signal")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_ecg["Time (ms)"], df_ecg["Processed ECG"], label="Processed ECG Signal", color='g')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Processed ECG Value")
    ax.set_title("Filtered & Denoised ECG Signal")
    ax.legend()
    st.pyplot(fig)

    # ✅ **Download Processed Data**
    st.download_button("Download Processed Data CSV", df_ecg.to_csv(index=False).encode("utf-8"), "processed_ecg.csv", "text/csv")

    st.subheader("✅ ECG Processing Complete!")
