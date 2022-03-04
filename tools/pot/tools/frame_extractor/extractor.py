# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cv2 as cv
import numpy as np
import video_loader


def sort_frame_indices_by_motion(video_file, frame_step=1, motion_quantile=0.95):
    """
     Compute motion on every frame from DataLoader and returns indices sorted by motion
    :param video_file: path to video file
    :param frame_step: number of frames to skip
    :param motion_quantile: quantile value to filter motion outliers on every image
    :return: list of indices sorted by motion
    """

    capture = cv.VideoCapture(video_file)
    frames_num = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    if frames_num == 1:
        return [0]

    prev = cv.cvtColor(capture.read()[1], cv.COLOR_BGR2GRAY)
    height, width = prev.shape[:2]
    req_height, req_width = 256, 256
    size = min(height, req_height), min(width, req_width)
    prev = cv.resize(prev, size)

    frame_motion_values = []
    for idx in range(1, frames_num):
        _, frame = capture.read()

        if idx % 500 == 0:
            print('Images processed {}'.format(idx))

        if idx % frame_step != 0 or frame is None:
            continue

        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.resize(frame, size)
        key_vectors = cv.calcOpticalFlowFarneback(
            prev, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        magnitudes = cv.magnitude(key_vectors[..., 0], key_vectors[..., 1])
        prev = frame

        frame_motion_values.append((idx, np.quantile(magnitudes, motion_quantile)))

    print('Key frames collecting finished')

    return [item[0] for item in sorted(frame_motion_values, key=lambda x: x[1], reverse=True)]


def extract_frames_and_make_dataset(
        video_file, output_dir, dataset_size, frame_step, ext='png'):
    """
    Extracts frames with the highest motion value and creates dataset in specified directory
    :param video_file: path to video file
    :param output_dir: directory to save extracted images as dataset
    :param dataset_size: number of images to extract
    :param frame_step: step to drop frames from video and exclude launching of algorithm for them
    :param ext: extension of images in dataset
    """

    frames_by_motion = sort_frame_indices_by_motion(video_file, frame_step)

    loader = video_loader.VideoLoader(video_file)
    if dataset_size is None:
        dataset_size = len(loader)
    if dataset_size > len(loader):
        raise RuntimeError('Number of images in output dataset should'
                           ' not be bigger than number of images in video')
    dataset_indices = sorted(frames_by_motion[:dataset_size])

    output_dir = Path(output_dir)
    for idx, dataset_idx in enumerate(dataset_indices):
        frame = loader[dataset_idx]
        cv.imwrite(str(output_dir / '{}.{}'.format(idx, ext)), frame)
