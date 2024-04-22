import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import spectrogram

# Load and normalize audio
x, fs = librosa.load("percepcja.wav", sr=None)
x = x - np.mean(x)
x = x / np.max(np.abs(x))

# Plot original signal
plt.figure(figsize=(10, 3))
plt.plot(x)
plt.ylim(-1, 1)
plt.title("Original Signal")
plt.show()

# Length of analysis window: 200 ms
win = int(fs * 0.2)

# Instantaneous energy
ste = np.convolve(x ** 2, np.ones(win), 'same')
ste = ste / np.max(ste)

# Square root version
str_ = np.sqrt(ste)

# Plot energy and square root energy
plt.figure(figsize=(10, 3))
plt.plot(x)
plt.plot(ste)
plt.plot(str_)
plt.ylim(0, 1)
plt.legend(["x", "STE", "STR"])
plt.title("Energy and Square Root Energy")
plt.show()


# second window of matlab notebook
plt.figure(figsize=(10, 3))
plt.plot(x)
plt.ylim(-1, 1)
plt.plot(0.2 * (ste > 0.05))
plt.plot(0.4 * (ste > 0.01))
plt.plot(0.6 * (str_ > 0.05))
plt.plot(0.8 * (str_ > 0.01))
plt.legend(["x", "STE > 0.05", "STE > 0.01", "STR > 0.05", "STR > 0.01"], loc="lower left")
plt.show()


# third window
x, fs = librosa.load("es.wav", sr=None)

# Define window size
win = int(fs * 0.025)

# Plot the entire signal and cropped fragments
plt.figure(figsize=(15, 3))
plt.subplot(1, 3, 1)
plt.plot(x)
plt.title("Entire Signal")
plt.ylim(-0.2, 0.2)

plt.subplot(1, 3, 2)
e_crop = x[10000:10000+win]
plt.plot(e_crop)
plt.title("Fragment E")
plt.xlim(0, win)
plt.ylim(-0.1, 0.1)

plt.subplot(1, 3, 3)
s_crop = x[25000:25000+win]
plt.plot(s_crop)
plt.title("Fragment S")
plt.xlim(0, win)
plt.ylim(-0.1, 0.1)

plt.tight_layout()
plt.show()

def zerocrossrate(x):
    zc = np.diff(np.sign(x))
    zcr = np.sum(zc != 0) / len(x)
    return zcr


# Define window size
win = int(fs * 0.025)

# Create figure
plt.figure(figsize=(10, 3))
plt.subplot(1, 1, 1)

# Plot spectrogram
f, t, Sxx = spectrogram(x, fs, nperseg=win, noverlap=win//2)
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(0, fs/2)
plt.colorbar(label='Intensity [dB]')
plt.title('Spectrogram')

plt.tight_layout()
plt.show()

print(zerocrossrate(e_crop))
print(zerocrossrate(s_crop))

# Define window size
win = int(fs * 0.025)

# Create x1 and x2
x1 = x
x2 = np.concatenate(([0], x[:-1]))  # Pad x2 with 0 at the beginning

# Calculate diff
diff = np.sign(x1) != np.sign(x2)

# Calculate zero-crossing rate (zcr)
zcr = np.convolve(diff.astype(float), np.ones(win) / win, "same")

# Plot normalized signal and zcr
plt.figure(figsize=(10, 3))
plt.plot(x / np.max(np.abs(x)), label='Normalized Signal')
plt.plot(zcr, label='Zero-Crossing Rate (ZCR)')
plt.ylim(-1, 1)
plt.legend()
plt.show()

from scipy.signal import find_peaks
from scipy.signal import correlate
import librosa

# Define window size
win = int(fs * 0.025)

# Define f_max and lag_min
f_max = 300
lag_min = int(fs / f_max)

# Crop fragments
e_crop = x[10000:10000+win]
s_crop = x[25000:25000+win]

# Create tiled layout
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)

# Cross-correlation and peak detection for fragment s_crop
c = correlate(s_crop, s_crop, mode='full')
lags = np.arange(-len(s_crop) + 1, len(s_crop))
c = c[lags > lag_min]
lags = lags[lags > lag_min]
peaks, _ = find_peaks(c, prominence=0.2, width=3)
plt.plot(lags, c)
plt.plot(lags[peaks], c[peaks], 'x')
plt.title('Fragment S Cross-correlation')
plt.xlabel('Lag')
plt.ylabel('Normalized Cross-correlation')

plt.subplot(1, 2, 2)

# Cross-correlation and peak detection for fragment e_crop
c = correlate(e_crop, e_crop, mode='full')
lags = np.arange(-len(e_crop) + 1, len(e_crop))
c = c[lags > lag_min]
lags = lags[lags > lag_min]
peaks, _ = find_peaks(c, prominence=0.2, width=3)
plt.plot(lags, c)
plt.plot(lags[peaks], c[peaks], 'x')
plt.title('Fragment E Cross-correlation')
plt.xlabel('Lag')
plt.ylabel('Normalized Cross-correlation')

plt.tight_layout()
plt.show()