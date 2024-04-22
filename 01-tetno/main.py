from typing import List
from matplotlib import pyplot as plt

from pulse import (
    read_frames_and_framerate_from_video,
    red_middle_pixel_values,
    red_pixel_means
)


def count_intersections(signal: List[float], constant_value: int) -> int:
    '''
    Number of signal intersections with a constant value (y = constant_val)
    '''
    intersections = 0

    for i in range(1, len(signal)):
        if (signal[i - 1] <= constant_value and signal[i] > constant_value) or \
           (signal[i - 1] >= constant_value and signal[i] < constant_value):
            intersections += 1

    return intersections


def count_falling_intersections_indices(signal: List[float], constant_value: int) -> List[int]:
    '''
    Number of signal intersections with a constant value (y = constant_val)
    where signal is falling, not rising
    '''
    intersection_indices = []

    for i in range(1, len(signal)):
        if (signal[i - 1] >= constant_value and signal[i] < constant_value):
            intersection_indices.append(i)

    return intersection_indices


def frames_per_beats(intersection_points: List[float]) -> float:
    '''
    Given indices of intersection points calculate average length
    of a beat in frames
    '''
    total = intersection_points[-1] - intersection_points[0]
    no_of_beats = len(intersection_points) - 1
    return total / no_of_beats


if __name__ == '__main__':
    video_path = "puls.mp4"
    frames, frame_rate = read_frames_and_framerate_from_video(video_path)

    frames = frames[300:]
    # frames shape H, W, Ch

    middle_red_pixels = red_middle_pixel_values(frames)
    red_pixel_mean = red_pixel_means(frames)

    plt.plot(middle_red_pixels)
    plt.title('Middle red pixel value')
    plt.savefig('middle_red_pixel_values.png')
    plt.show()
    plt.clf()

    plt.plot(red_pixel_mean)
    plt.title('Mean red pixel value')
    plt.savefig('mean_red_pixel_values.png')
    plt.show()

    intersection_points = count_falling_intersections_indices(red_pixel_mean, 76)

    # BPM calculation
    frames_per_beat_value = frames_per_beats(intersection_points)

    cycles = len(frames) / frames_per_beat_value
    video_len_minutes = len(frames)/(frame_rate * 60)
     
    bpm = cycles / video_len_minutes
    print(bpm)

    # resolution calculation
    intersection_points[-1] = intersection_points[-1] + 1

    frames_per_beat_value = frames_per_beats(intersection_points)

    cycles = len(frames) / frames_per_beat_value
    video_len_minutes = len(frames)/(frame_rate * 60)
     
    bpm_resolution = cycles / video_len_minutes
    print(f'Resolution: {bpm - bpm_resolution}')
