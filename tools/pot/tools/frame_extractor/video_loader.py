# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from cv2 import VideoCapture, CAP_PROP_FRAME_COUNT


class VideoLoader:

    def __init__(self, video_file):
        self._frame_pointer = -1

        self._video_file = video_file
        self._capture = VideoCapture(video_file)

    def __getitem__(self, idx):

        if idx >= len(self) or idx < 0:
            raise IndexError

        if idx < self._frame_pointer:
            self._capture = VideoCapture(self._video_file)
            self._frame_pointer = -1

        image = None
        while self._frame_pointer < idx:
            _, image = self._capture.read()
            self._frame_pointer += 1

        if image is not None:
            return image

    def __len__(self):
        return int(self._capture.get(CAP_PROP_FRAME_COUNT))
