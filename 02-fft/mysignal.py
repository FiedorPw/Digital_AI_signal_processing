'''
Functions dedicated to generating signals
'''
from typing import List
import numpy as np


def generate_signal(amplitudes: list[float],
                    frequencies: list[float],
                    phase_shifts: list[float],
                    sampling: float,
                    num_samples: int) -> List[float]:
    '''
    Generate a signal made of multiple sine waves.

    :param amplitudes: list of amplitudes of sine waves
    :param frequencies: list of frequencies of sine waves
    :param phase_shifts: list of phase shifts of sine waves
    :param sampling: sampling frequencies in herts
    :param num_samples: number of samples to return
    :return: sampled values of the signal
    '''
    assert len(amplitudes) == len(frequencies) == len(phase_shifts)

    time_moments = [i / sampling for i in range(num_samples)]

    signal_f = lambda x: sum([a * np.cos(2 * np.pi * f * x + p) for a, f, p in zip(amplitudes, frequencies, phase_shifts)])

    return [signal_f(x) for x in time_moments]


def noise_signal_normal(signal: List[float]) -> List[float]:
    '''
    Add noise to a given signal with normal distribution.

    :param signal: input signal
    :return: signal noised with normal distribution noise
    '''
    return [x + np.random.normal() for x in signal]
