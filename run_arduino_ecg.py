import serial
import csv
import time

# Set your Arduino's serial port (change COMx or /dev/ttyUSBx accordingly)
SERIAL_PORT = "COM2"  # Windows Example: "COM3", Linux/macOS Example: "/dev/ttyUSB0"
BAUD_RATE = 115200
OUTPUT_FILE = "ecg_data.csv"

# Open Serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Wait for Arduino to initialize
time.sleep(2)

# Open CSV file to save data
with open(OUTPUT_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time (ms)", "Raw ECG"])  # CSV Header

    try:
        while True:
            line = ser.readline().decode("utf-8").strip()  # Read serial data
            if line and "," in line:
                print(line)  # Print to console
                writer.writerow(line.split(","))  # Save to CSV
    except KeyboardInterrupt:
        print("\nData logging stopped.")
        ser.close()
