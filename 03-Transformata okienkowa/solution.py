import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, gaussian, find_peaks
from scipy.fft import fft

# Signal generation
t = np.arange(0, 1, 0.001)
sig_1 = 0.5 * np.sin(6 * np.pi * t) + 0.1 * np.random.randn(1000)
sig_2 = np.sign(0.5 * np.sin(6 * np.pi * t)) + 0.1 * np.random.randn(1000)

def twoplots(t, s1, s2):
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1)
    plt.plot(t, s1)
    plt.subplot(1, 2, 2)
    plt.plot(t, s2)
    plt.show()

twoplots(t, sig_1, sig_2)


# Moving average filter
f3 = np.ones(3) / 3
c3_1 = convolve(sig_1, f3, mode='same')
c3_2 = convolve(sig_2, f3, mode='same')
twoplots(t, c3_1, c3_2)

f15 = np.ones(15) / 15
c15_1 = convolve(sig_1, f15, mode='same')
c15_2 = convolve(sig_2, f15, mode='same')
twoplots(t, c15_1, c15_2)

# Gaussian filter
def fspecial_gaussian(size, sigma):
    """Approximate MATLAB's 'fspecial' function for a Gaussian kernel."""
    x = np.arange(-(size-1)/2, (size+1)/2)
    gauss = np.exp(-x**2 / (2*sigma**2))
    return gauss / gauss.sum()

g3 = fspecial_gaussian(3, 1)
cg3_1 = convolve(sig_1, g3, mode='same')
cg3_2 = convolve(sig_2, g3, mode='same')
twoplots(t, cg3_1, cg3_2)

g15 = fspecial_gaussian(15, 3)
cg15_1 = convolve(sig_1, g15, mode='same')
cg15_2 = convolve(sig_2, g15, mode='same')
twoplots(t, cg15_1, cg15_2)


# Edge detection filter
e3 = np.array([1, 0, -1])
ce3_1 = convolve(sig_1, e3, mode='same')
ce3_2 = convolve(sig_2, e3, mode='same')
twoplots(t, ce3_1, ce3_2)


# Autocorrelation
from scipy.signal import correlate

def autocorr(sig):
    result = correlate(sig, sig, mode='full')
    return result[result.size // 2:]

r1 = autocorr(sig_1)
r2 = autocorr(sig_2)
lags = np.arange(len(r1))

twoplots(lags, r1, r2)

# Finding the base frequency
pks, locs = find_peaks(r1, distance=10, prominence=20)
fs = 1000  # Sampling frequency
lag_s = locs[0] / fs
freq = 1 / lag_s
print(f"Base frequency: {freq:.4f} Hz")


def filtstat(f):
    nfft = 1000
    fs = 1000

    f_ex = np.zeros(nfft)
    f_ex[:len(f)] = f

    y = fft(f_ex)
    f_base = np.linspace(0, fs / 2, int(nfft / 2) + 1)
    amp = np.abs(y[:int(nfft / 2) + 1])
    phase = np.angle(y[:int(nfft / 2) + 1])

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1)
    plt.plot(f_base, amp)
    plt.title('Amplitude Response')
    plt.subplot(1, 2, 2)
    plt.plot(f_base, phase)
    plt.title('Phase Response')
    plt.show()

# Example usage:
# filtstat(f3)

#guasian filter ze scipi wygładzić sygnał wejsciowy
# wynik autokorelacji czyli np 23 powtórzenia na 30
#rozdzielczość zwiekszyc albo zmniejszyć o 1
#zobaczyć 10 peak a nie 1 i podzielić na 10