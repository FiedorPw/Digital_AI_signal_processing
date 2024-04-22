import numpy as np
import matplotlib.pyplot as plt
import librosa

# Load audio file
x, fs = librosa.load("percepcja.wav", sr=None)

# Normalize audio
x = x / np.max(np.abs(x))

# Label segments
def label(x, fs):
    # Your label function implementation here
    pass

lbls = label(x, fs)

# Calculate pitch window
def pitchwin(x, fs, duration):
    # Your pitchwin function implementation here
    pass

f0 = pitchwin(x, fs, fs * 0.1)
f0[lbls != 2] = 0

# Segregate into different types
silence = np.copy(x)
silence[lbls != 0] = 0

unvoiced = np.copy(x)
unvoiced[lbls != 1] = 0

voiced = np.copy(x)
voiced[lbls != 2] = 0

# Plot signals
plt.figure()
plt.title("Signal Segmentation")
plt.plot(unvoiced, label="Unvoiced")
plt.plot(voiced, label="Voiced")
plt.plot(silence, label="Silence")
plt.ylim(-1, 1)
plt.legend()
plt.show()

plt.figure()
plt.title("Pitch Contour")
plt.plot(f0)
plt.show()
