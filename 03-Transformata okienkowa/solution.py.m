import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(filename, window_size):
    # Load the audio file
    # `sr=None` ensures the original sampling rate is preserved
    x, fs = librosa.load(filename, sr=None)

    win_len = window_size
    win_overlap = 256
    nfft = window_size

    # Compute the Short-Time Fourier Transform (STFT)
    # Note: `n_fft` and `hop_length` are the equivalent parameters to `nfft` and `win_overlap`
    D = librosa.stft(x, n_fft=nfft, hop_length=win_overlap, win_length=win_len)

    # Convert amplitude to decibels
    D_db = librosa.amplitude_to_db(abs(D), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D_db, sr=fs, x_axis='time', y_axis='log', hop_length=win_overlap)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()


def plot_frequencies(filename, window_size):
    # Load the audio file
    x, fs = librosa.load(filename, sr=None)  # sr=None to keep original sampling rate

    # Parameters
    win_len = window_size  # Window size
    win_overlap = 256  # Overlap between windows
    nfft = window_size  # Number of FFT points

    # Compute the Short-Time Fourier Transform (STFT)
    D = librosa.stft(x, n_fft=nfft, hop_length=win_overlap, win_length=win_len)

    # Calculate amplitude
    A = np.abs(D) / nfft

    # Convert frequency and time indices to physical units
    f = librosa.fft_frequencies(sr=fs, n_fft=nfft)
    t = librosa.frames_to_time(np.arange(A.shape[1]), sr=fs, hop_length=win_overlap)

    # Plot amplitude of selected windows
    plt.figure()
    selected_windows = [10, 20, 30]
    for i in selected_windows:
        plt.plot(f, A[:, i], label=f'{t[i]:.3f} s')

    plt.xlabel('Frequency (Hz)')
    plt.legend(title='Window Center Time')
    plt.title('Amplitude Spectrum of Selected Windows')

    # Plot time course of selected frequencies
    plt.figure()
    selected_frequencies = [10, 30, 50]
    for i in selected_frequencies:
        plt.plot(t, A[i, :], label=f'{f[i]:.1f} Hz')

    plt.xlabel('Time (s)')
    plt.legend(title='Frequency')
    plt.title('Time Course of Selected Frequencies')

    plt.show()


def get_frequency_amplitudes(filename, frequencies):
    y, sr = librosa.load(filename, sr=None)

    # Compute the Short-Time Fourier Transform (STFT)
    D = np.abs(librosa.stft(y))

    # Get the frequencies for each bin in the STFT
    fft_frequencies = librosa.fft_frequencies(sr=sr)

    # Initialize a dictionary to store the average amplitudes
    amplitude_dict = {}

    for freq in frequencies:
        # Find the closest match for the requested frequency in the FFT bin frequencies
        index = np.argmin(np.abs(fft_frequencies - freq))

        # Compute the average amplitude for this frequency
        amplitude = np.mean(D[index, :])

        # Store the average amplitude in the dictionary
        amplitude_dict[freq] = amplitude

    return amplitude_dict


def find_active_segments(filename, threshold, frame_length=2048, hop_length=512):
    y, sr = librosa.load(filename, sr=None)
    # Compute the short-term energy of the audio signal
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length)) ** 2
    energy = np.sum(S, axis=0)

    # Normalize energy
    energy = librosa.util.normalize(energy, axis=0)

    # Convert energy signal to binary (active/not active) based on threshold
    active_frames = np.where(energy > threshold, 1, 0)

    # Identify segment boundaries
    edges = np.diff(active_frames)
    start_frames = np.where(edges > 0)[0] + 1  # +1 to compensate for the diff operation
    stop_frames = np.where(edges < 0)[0] + 1

    # Handle case where signal starts/ends with activity
    if active_frames[0] == 1:
        start_frames = np.insert(start_frames, 0, 0)
    if active_frames[-1] == 1:
        stop_frames = np.append(stop_frames, len(active_frames) - 1)

    # Convert frame indices to time
    start_times = librosa.frames_to_time(start_frames, sr=sr, hop_length=hop_length)
    stop_times = librosa.frames_to_time(stop_frames, sr=sr, hop_length=hop_length)

    # Pair start and stop times
    segments = list(zip(start_times, stop_times))

    return segments


def get_frequency_amplitudes_at_time(filename, frequencies, time):
    y, sr = librosa.load(filename, sr=None)

    # Compute the Short-Time Fourier Transform (STFT)
    D = np.abs(librosa.stft(y))

    # Calculate the time for each frame in the STFT
    times = librosa.frames_to_time(range(D.shape[1]), sr=sr)

    # Find the index of the frame that is closest to the specified time
    time_index = np.argmin(np.abs(times - time))

    # Get the frequencies for each bin in the STFT
    fft_frequencies = librosa.fft_frequencies(sr=sr)

    # Initialize a dictionary to store the amplitudes
    amplitude_dict = {}

    for freq in frequencies:
        # Find the closest match for the requested frequency in the FFT bin frequencies
        freq_index = np.argmin(np.abs(fft_frequencies - freq))

        # Get the amplitude for this frequency at the specified time
        amplitude = D[freq_index, time_index]

        # Store the amplitude in the dictionary
        amplitude_dict[freq] = amplitude

    return amplitude_dict


def dtmf(filename, threshold):
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "0", "#"]
    frequencies1 = [1200, 1336, 1477]  # Example frequencies in Hz
    frequencies2 = [697, 770, 852, 941]
    # moments with signal
    active_segments = find_active_segments('dtmf.wav', threshold)
    x_index = []
    y_index = []
    decoded_numbers = []

    for segment in active_segments:
        # middle of the active
        time = (segment[0] + segment[1]) / 2

        x = get_frequency_amplitudes_at_time(filename, frequencies1, time)
        y = get_frequency_amplitudes_at_time(filename, frequencies2, time)

        print("amplitudy dla f", x, y)

        x = max(x, key=x.get)
        y = max(y, key=y.get)

        print("największa", x, y)

        for i in range(len(frequencies1)):
            if x == frequencies1[i]:
                x_index.append(i)
        print(x_index)

        for i in range(len(frequencies2)):
            if y == frequencies2[i]:
                y_index.append(i)
        print(y_index)

    for i in range(len(x_index)):
        index = x_index[i] + 3 * y_index[i]
        decoded_numbers.append(labels[index])
    print(decoded_numbers)



# frequencies1 = [1200, 1336, 1477]
# frequencies2 = [697, 770, 852, 941]

# plot_spectrogram('dtmf.wav', 512)
# plot_spectrogram('lewo.wav', 512)
# plot_frequencies('dtmf.wav', 512)
# amplitudes = get_frequency_amplitudes('dtmf.wav', frequencies1)
# print(amplitudes)

# print(get_frequency_amplitudes_at_time('dtmf.wav', frequencies2, 2.1))

# threshold = 0.02

# print(active_segments)

dtmf('dtmf.wav', 0.02)
