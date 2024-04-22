'''
Functions used to read video from mp4 and read red pixels
'''
from typing import List
import numpy as np
import cv2


def read_frames_and_framerate_from_video(video_path: str) -> List[np.ndarray]:
    video = cv2.VideoCapture(video_path)
    
    frames = []

    frame_rate = int(video.get(cv2.CAP_PROP_FPS))

    # Read until video is completed
    while video.isOpened():
        # Read a frame from the video
        ret, frame = video.read()

        # If frame is read correctly, append it to the list
        if ret:
            frames.append(frame)
        else:
            break

    video.release()

    return frames, frame_rate


def red_middle_pixel_values(frames: List[np.ndarray]) -> List[float]:
    '''
    For a given list of images in np.ndarrays return list of
    values for red channel's middle pixel
    '''
    values = []
    frame_shape = frames[0].shape
    middle_h, middle_w = (frame_shape[0] // 2, frame_shape[1] // 2)
    for frame in frames:
        values.append(frame[middle_h][middle_w][2])

    return values


def red_pixel_means(frames: List[np.ndarray]) -> List[float]:
    '''
    For a given list of images in np.ndarrays return list of
    values for red channel's middle pixel
    '''
    values = []
    for frame in frames:
        values.append(np.mean(frame[:][:][2]))

    return values
