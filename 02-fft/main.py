from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np

from mysignal import (generate_signal,noise_signal_normal)
from fourier import (
    decompose_signal,
    n_strongest_frequencies,
    n_strongest_frequencies_resolution
)
from pulse import (
    read_frames_and_framerate_from_video,
    red_pixel_means
)

if __name__ == '__main__':
    # parameters of the signal
    amplitudes = [1.0, 0.4, 0.8]
    frequencies = [15, 27, 83]
    phase_shifts = [0, -1 * np.pi / 3, np.pi / 7]

    # parameters of the sampling of the signal
    num_samples = 10000
    sampling_freq = 2000
    len_signal_seconds = num_samples/sampling_freq

    # generate the signal
    signal = generate_signal(
        amplitudes,
        frequencies,
        phase_shifts,
        sampling_freq,
        num_samples
    )

    # plot for debug purposes and save
    plt.figure()
    plt.plot(signal[:1000])
    plt.title('Generated signal')
    plt.savefig('generated_signal.png')

    # plot amplitude and phase and save
    amplitudes, phases = decompose_signal(signal)
    plt.figure()
    plt.plot(amplitudes)
    plt.title('Amplitude')
    plt.savefig('amplitude.png')
    plt.figure()
    plt.plot(phases)
    plt.title('Phase')
    plt.savefig('phase.png')

    # print recovered frequencies
    print(n_strongest_frequencies(signal, len_signal_seconds, 3))

    # add noise to the signal
    noised_signal = noise_signal_normal(signal)

    # plot both clean and noised signal and save
    plt.figure()
    plt.plot(noised_signal[:200])
    plt.plot(signal[:200])
    plt.title('Noised signal')
    plt.savefig('noised_signal.png')

    # print recovered frequencies
    print(n_strongest_frequencies(noised_signal, len_signal_seconds, 3))

    # read red pixels from video
    video_path = "puls.mp4"
    frames, frame_rate = read_frames_and_framerate_from_video(video_path)

    frames = frames[300:]
    len_video_seconds = len(frames) / frame_rate

    red_pixel_mean = red_pixel_means(frames)

    # normalize red pixel mean
    mean = np.mean(red_pixel_mean)
    red_pixel_mean = [x - mean for x in red_pixel_mean]

    # plot pulse for debug and save
    plt.figure()
    plt.plot(red_pixel_mean)
    plt.title('Pulse normalized')
    plt.savefig('pulse_normalized.png')

    # plot decomposition of read pixel mean and save
    amplitudes, _ = decompose_signal(red_pixel_mean)
    plt.figure()
    plt.plot(amplitudes)
    plt.title('Amplitude of Red Pixel Mean')
    plt.savefig('amplitude_red_pixel_mean.png')

    bpm_real = n_strongest_frequencies(red_pixel_mean, len_video_seconds, 1)[0] * 60
    bpm_resolution = n_strongest_frequencies_resolution(red_pixel_mean, len_video_seconds, 1)[0] * 60
    resolution = bpm_resolution - bpm_real

    print('Real BPM: ', bpm_real)
    print('Measurement resolution ', resolution)
