import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.ndimage
from scipy.signal import find_peaks
import time

# Configuration for PyAudio
format1 = pyaudio.paInt16  # 16-bit PCM audio format
channels1 = 1  # Mono channel
rate1 = 44100  # Sample rate
chunk1 = 1024  # Number of frames per buffer
record_duration = 2  # Duration to record after detecting a peak in seconds

# Initialize PyAudio
pya = pyaudio.PyAudio()

# Setup matplotlib
fig, ax = plt.subplots()
x = np.linspace(0, rate1 // 2, chunk1 // 2)
line, = ax.plot(x, np.zeros(chunk1 // 2))
peak_points, = ax.plot([], [], "x", color='red')  # Peaks will be shown as red crosses
ax.set_xlim(0, rate1 / 2)
ax.set_ylim(0, 5000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Live Fourier Transform')

# Start the audio stream
stream = pya.open(format=format1, channels=channels1, rate=rate1, input=True, frames_per_buffer=chunk1, input_device_index=1)

# Variables for dynamic threshold calculation and recording post peak detection
calibration_time = 3  # seconds for calibration
samples_for_calibration = (rate1 // chunk1) * calibration_time
total_magnitude = 0
calibration_samples = 0
min_magnitude_threshold = 0  # Initial threshold, will be set dynamically
peak_detected = False
start_time = 0
recorded_peaks = []  # Array to store peaks after detection

def update(frame):
    global total_magnitude, calibration_samples, min_magnitude_threshold, peak_detected, start_time, recorded_peaks

    # Read raw audio data
    data = stream.read(chunk1, exception_on_overflow=False)
    # Convert audio to a NumPy array
    audioData = np.frombuffer(data, dtype=np.int16)
    # Apply high pass filter
    filteredAudioData = highPassFilt(audioData, 1000, rate1)
    # Apply Fast Fourier Transform
    fastFT_data = np.fft.fft(filteredAudioData)
    # Get frequency magnitudes
    magnitudes = np.abs(fastFT_data)[:chunk1 // 2] * 2 / chunk1  # Normalize and take the first half
    # Smooth the magnitudes using a Gaussian filter
    smooth_magnitudes = scipy.ndimage.gaussian_filter(magnitudes, sigma=1.2)

    if calibration_samples < samples_for_calibration:
        total_magnitude += np.mean(smooth_magnitudes)
        calibration_samples += 1
        if calibration_samples == samples_for_calibration:
            average_magnitude = total_magnitude / samples_for_calibration
            min_magnitude_threshold = (average_magnitude * 5) * 10  # Use to find right cutoff for background noise
            print(f"Calibration complete. Dynamic threshold set to: {min_magnitude_threshold}")
    else:
        # Detect peaks with the dynamically set minimum magnitude threshold
        peaks, properties = find_peaks(smooth_magnitudes, height=min_magnitude_threshold)
        # Filter out peaks above 2700 Hz
        valid_indices = [i for i in range(len(peaks)) if x[peaks[i]] <= 2700]
        valid_peaks = peaks[valid_indices]
        valid_properties = properties['peak_heights'][valid_indices]

        if len(valid_peaks) > 0 and not peak_detected:
            peak_detected = True
            start_time = time.time()
            recorded_peaks = []  # Reset the array to start new recording
            print("Peak detected. Starting to record peaks.")

        if peak_detected:
            # Record peaks for 2 seconds after the first peak
            if time.time() - start_time <= record_duration:
                recorded_peaks.append((x[valid_peaks], valid_properties))
            else:
                peak_detected = False  # Reset after recording is complete
                print("Recording complete. Peaks have been stored.")
                print("Recorded Peaks:")
                gestureRecognizer(recorded_peaks)  # Call gestureRecognizer here after recording is done

        # Update the plot lines
        line.set_ydata(smooth_magnitudes)
        peak_points.set_data(x[valid_peaks], valid_properties)

    return line, peak_points,


def highPassFilt(audioDat, cutoffFr, rate2):
    alpha = 1 / (1 + np.tan(np.pi * cutoffFr / rate2))
    beta = alpha - 1
    output_signal = np.zeros_like(audioDat, dtype=np.float64)
    output_signal[0] = float(audioDat[0])  # Start with the first sample in float
    for n in range(1, len(audioDat)):
        filtered_value = alpha * (float(audioDat[n]) - float(audioDat[n-1])) + beta * output_signal[n-1]
        output_signal[n] = filtered_value
    return output_signal

def gestureRecognizer(recordedPeaks):
    # Filter out empty values or where frequency is over 2600 Hz
    recordedPeaks = [(peak, height) for peak, height in recordedPeaks if peak.size > 0 and height.size > 0 and peak[0] <= 2600]
    numPeak = len(recordedPeaks)
    print("Number of valid peaks:", numPeak)
    # Print test data
    #for peak, height in recordedPeaks:
        # print("Frequency: {} Hz, Magnitude: {}".format(peak[0], height[0]))

    # Divide numPeak by 3
    section_size = numPeak // 3
    remainder = numPeak % 3

    # split into 3 sections each recognized individually
    section1 = []
    section2 = []
    section3 = []
    start_index = 0
    for i in range(3):
        if remainder > 0:
            end_index = start_index + section_size + 1
            remainder -= 1
        else:
            end_index = start_index + section_size
        if i == 0:
            section1 = recordedPeaks[start_index:end_index]
        elif i == 1:
            section2 = recordedPeaks[start_index:end_index]
        elif i == 2:
            section3 = recordedPeaks[start_index:end_index]
        start_index = end_index
    #process the 3 sections

    #print("Section 1:")
    #for peak, height in section1:
    #    print(f"  Frequency: {peak[0]} Hz, Magnitude: {height[0]}")
    
    #print("Section 2:")
    #for peak, height in section2:
    #    print(f"  Frequency: {peak[0]} Hz, Magnitude: {height[0]}")

    #print("Section 3:")
    #for peak, height in section3:
    #    print(f"  Frequency: {peak[0]} Hz, Magnitude: {height[0]}")

    total_freq_section1 = sum(peak[0][0] for peak in section1)
    total_freq_section2 = sum(peak[0][0] for peak in section2)
    total_freq_section3 = sum(peak[0][0] for peak in section3)

    #print("Total sec 1 " , total_freq_section1)
    #print("Total sec 2 " , total_freq_section2)
    #print("Total sec 3 " , total_freq_section3)

    sec1Avg = total_freq_section1 / len(section1)
    sec2Avg = total_freq_section2 / len(section2)
    sec3Avg = total_freq_section3 / len(section3)

    print("avg sec 1 " , sec1Avg)
    print("avg sec 2 " , sec2Avg)
    print("avg sec 3 " , sec3Avg)

    #rank low mid high

    diff12 = abs(sec1Avg - sec2Avg)
    diff23 = abs(sec2Avg - sec3Avg)
    diff31 = abs(sec3Avg - sec1Avg)

    # All differences within 50 Hz
    if diff12 <= 50 and diff23 <= 50 and diff31 <= 50:
        rankings = {'sec1': 'mid', 'sec2': 'mid', 'sec3': 'mid'}
    else:
        if diff12 <= 50:
            if diff23 > 50 and diff31 > 50:
                rankings = {'sec1': 'low', 'sec2': 'low', 'sec3': 'high'} if sec3Avg > sec1Avg else {'sec1': 'high', 'sec2': 'high', 'sec3': 'low'}
        elif diff23 <= 50:
            if diff12 > 50 and diff31 > 50:
                rankings = {'sec2': 'low', 'sec3': 'low', 'sec1': 'high'} if sec1Avg > sec2Avg else {'sec2': 'high', 'sec3': 'high', 'sec1': 'low'}
        elif diff31 <= 50:
            if diff12 > 50 and diff23 > 50:
                rankings = {'sec3': 'low', 'sec1': 'low', 'sec2': 'high'} if sec2Avg > sec3Avg else {'sec3': 'high', 'sec1': 'high', 'sec2': 'low'}
        else:
            rankings = {
                'sec1': 'high' if sec1Avg > sec2Avg and sec1Avg > sec3Avg else 'mid' if sec1Avg > min(sec2Avg, sec3Avg) else 'low',
                'sec2': 'high' if sec2Avg > sec1Avg and sec2Avg > sec3Avg else 'mid' if sec2Avg > min(sec1Avg, sec3Avg) else 'low',
                'sec3': 'high' if sec3Avg > sec1Avg and sec3Avg > sec2Avg else 'mid' if sec3Avg > min(sec1Avg, sec2Avg) else 'low'
            }
    print("Rankings:", rankings)




# Create the animation object
ani = FuncAnimation(fig, update, blit=True, interval=20, save_count=2000)  # Avoid unbounded memory use

# Display the plot and start recording
print("Audio recording, exit with [ctrl]+c ")
try:
    plt.show()
except KeyboardInterrupt:
    print("Recording Stopped")
finally:
    # Close the stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    pya.terminate()
