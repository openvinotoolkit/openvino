# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from glob import glob
from os.path import isfile, isdir, isabs
from pathlib import Path

import numpy as np
import cv2 as cv

from openvino.runtime import Layout # pylint: disable=E0611,E0401
from openvino.tools.pot.utils.logger import get_logger

logger = get_logger(__name__)


def crop(image, central_fraction):

    height, width = image.shape[:2]

    dst_height = int(height * central_fraction)
    dst_width = int(width * central_fraction)

    if height < dst_height or width < dst_width:
        resized = np.array([width, height])
        if resized[0] < dst_width:
            resized *= dst_width / resized[0]
        if resized[1] < dst_height:
            resized *= dst_height / resized[1]
        image = cv.resize(image, tuple(np.ceil(resized).astype(int)))

    start_height = (height - dst_height) // 2
    start_width = (width - dst_width) // 2
    return image[start_height:start_height + dst_height, start_width:start_width + dst_width]


def prepare_image(image, layout, dst_shape=None, central_fraction=None, grayscale=False):
    if central_fraction:
        image = crop(image, central_fraction)

    if dst_shape:
        image = cv.resize(image, dst_shape[::-1])

    if grayscale:
        image = np.expand_dims(image, 2)

    if layout == Layout('NCHW') or layout == Layout('CHW'):
        return image.transpose(2, 0, 1)

    return image


def collect_img_files(data_source):
    data_source = str(data_source)
    directory = Path(data_source if isdir(data_source) else os.path.dirname(data_source))

    def get_images_from_file(file_name):
        image_files = []
        with open(file_name) as file:
            for line in file.readlines():
                line = line.rstrip('\n')
                if isabs(line):
                    image_files.append(line)
                else:
                    image_files.append(str(directory.joinpath(line)))
        return image_files

    if isfile(data_source):
        if not data_source.endswith('.txt'):
            raise Exception('Only .txt files or directories can be set as data_source')
        return get_images_from_file(data_source)

    file_names = []
    all_files_in_dir = os.listdir(data_source)\
        if isdir(data_source) else glob(data_source)

    for name in all_files_in_dir:
        file = str(directory.joinpath(name))
        if cv.haveImageReader(file):
            file_names.append(file)

    return sorted(file_names)
