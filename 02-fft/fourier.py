'''
Functions dedicated to fourier decomposition of a signal
'''
from typing import Tuple, List, Any
from matplotlib import pyplot as plt
import numpy as np


def half(collection: List[Any]) -> List[Any]:
    '''
    Return the first half of the collection

    :param collection: colletion to half
    :return: the first half of the collection
    '''
    len_collection = len(collection)
    return collection[:(len_collection//2)]


def decompose_signal(signal: List[float]) -> Tuple[List[float], List[float]]:
    '''
    Decomposes a signal using fast fourier transformation

    :param signal: signal to decompose
    :return: signal amplitudes and phases
    '''
    signal_fourier = np.fft.fft(signal)
    amplitudes = np.absolute(signal_fourier) * 2 / len(signal_fourier)
    phases = np.angle(signal_fourier)

    return half(amplitudes), phases



def n_strongest_frequencies(signal: List[float],
                            len_signal_seconds: float, 
                            num_maxes: int) -> List[float]:
    '''
    Get n strongest frequencies in a signal

    :param signal: signal to decompose
    :param len_signal_seconds: length of signal in seconds
    :param num_maxes: number of strongest frequencies to find
    :return: values of n strongest frequencies
    '''
    amplitudes, _ = decompose_signal(signal)

    strongest_amplitudes = list(reversed(sorted(amplitudes)))[:num_maxes]

    return [
        i / len_signal_seconds for i, a in enumerate(amplitudes)
        if a in strongest_amplitudes
    ]


def n_strongest_frequencies_resolution(signal: List[float],
                                       len_signal_seconds: float, 
                                       num_maxes: int) -> List[float]:
    '''
    Get n strongest frequencies in a signal

    :param signal: signal to decompose
    :param len_signal_seconds: length of signal in seconds
    :param num_maxes: number of strongest frequencies to find
    :return: values of n strongest frequencies
    '''
    amplitudes, _ = decompose_signal(signal)

    strongest_amplitudes = list(reversed(sorted(amplitudes)))[:num_maxes]

    return [
        (i + 1) / len_signal_seconds for i, a in enumerate(amplitudes)
        if a in strongest_amplitudes
    ]
