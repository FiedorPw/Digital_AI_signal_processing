import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import convolve, freqz


def reverse_sinc_filter(fc, N, fs, window='hamming'):
    t = np.arange(-N // 2, N // 2 + 1) / fs
    sinc_function = np.sinc(2 * fc * t)

    if window == 'hamming':
        w = np.hamming(N + 1)
    elif window == 'blackman':
        w = np.blackman(N + 1)
    elif window == 'hanning':
        w = np.hanning(N + 1)
    else:
        raise ValueError("Unknown window type")

    filter_kernel = sinc_function * w
    filter_kernel /= np.sum(filter_kernel)  # Normalization
    # obrót
    reversed_signal = []
    for sample in filter_kernel:
        if sample != filter_kernel[filter_kernel.shape[0] // 2]:
            reversed_signal.append(-sample)
        else:
            sample += 1
            reversed_signal.append(sample)

    reversed_signal = np.array(reversed_signal)

    return reversed_signal


def sinc_filter(fc, N, fs, window='hamming'):
    t = np.arange(-N // 2, N // 2 + 1) / fs
    sinc_function = np.sinc(2 * fc * t)

    if window == 'hamming':
        w = np.hamming(N + 1)
    elif window == 'blackman':
        w = np.blackman(N + 1)
    elif window == 'hanning':
        w = np.hanning(N + 1)
    else:
        raise ValueError("Unknown window type")

    filter_kernel = sinc_function * w
    filter_kernel /= np.sum(filter_kernel)  # Normalization
    return filter_kernel

def create_high_pass_from_low_pass(low_pass_kernel):
    high_pass_kernel = -low_pass_kernel  # Invert the sign of all coefficients
    center_index = len(high_pass_kernel) // 2  # Find the index of the center coefficient
    high_pass_kernel[center_index] += 1  # Add 1 to the center coefficient
    return high_pass_kernel

def apply_filter(signal, filter_kernel):
    # Apply filter to each channel if stereo, else just apply to mono signal
    if signal.ndim == 1:  # Mono
        return convolve(signal, filter_kernel, mode='same')
    elif signal.ndim == 2:  # Stereo
        filtered_signal = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            filtered_signal[:, i] = convolve(signal[:, i], filter_kernel, mode='same')
        return filtered_signal


def filtstat(b, a, fs):
    w, h = freqz(b, a)
    plt.figure(figsize=(12, 3))
    plt.subplot(121)
    plt.plot(w * 0.5 * fs / np.pi, np.abs(h), 'b')
    plt.title('Frequency response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.subplot(122)
    plt.plot(w * 0.5 * fs / np.pi, np.angle(h), 'b')
    plt.title('Phase response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [radians]')
    plt.grid()
    plt.show()


# Load WAV file
signal, fs = sf.read('noised.wav')  # Replace 'path_to_your_wav_file.wav' with your file path

# Filter parameters
fc_max = 3200  # Cutoff frequency as a fraction of the sampling rate
fc_min = 900  # Cutoff frequency as a fraction of the sampling rate
N = 1001  # Number of samples in the filter

# Generate filter kernel
low_pass_kernel = sinc_filter(fc_max, N, fs, window='blackman')
# Generate low pass with other frequency and then reverse
kernel = sinc_filter(fc_min, N, fs, window='blackman')
high_pass_kernel = create_high_pass_from_low_pass(kernel)
# Apply filter
# print((filter_kernel))
# print((reverse_filter_kernel))

filtered_signal = apply_filter(signal, low_pass_kernel)
# reverse
filtered_signal = apply_filter(filtered_signal, high_pass_kernel)

# Plot the spectrogram of the original signal
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.specgram(signal, NFFT=1024, Fs=fs)
plt.title('Spectrogram of Original Signal')
plt.xlabel('Time')
plt.ylabel('Frequency')

# Plot the spectrogram of the filtered signal
plt.subplot(2, 1, 2)
plt.specgram(filtered_signal, NFFT=1024, Fs=fs)
plt.title('Spectrogram of Filtered Signal')
plt.xlabel('Time')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Display filter characteristics
# Assuming a and b are filter coefficients, for FIR filters b is the filter_kernel and a is 1
filtstat(low_pass_kernel, [1], fs)
filtstat(high_pass_kernel, [1], fs)

# Save the filtered signal to a new WAV file if needed
sf.write('filtered_signal.wav', filtered_signal, fs)
