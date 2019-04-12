"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging
import argparse
import os
import cv2
import numpy as np
import sys

from glob import glob
from random import choice
from datetime import datetime
from fnmatch import fnmatch

from .constants import *

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger('BenchmarkApp')


def validate_args(args):
    if args.number_iterations is not None and args.number_iterations < 0:
        raise Exception("Number of iterations should be positive (invalid -niter option value)")
    if args.number_infer_requests < 0:
        raise Exception("Number of inference requests should be positive (invalid -nireq option value)")
    if not fnmatch(args.path_to_model, XML_EXTENSION_PATTERN):
        raise Exception('Path {} is not xml file.')


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help=HELP_MESSAGES["HELP"])
    args.add_argument('-i', '--path_to_images', type=str, required=True, help=HELP_MESSAGES['IMAGE_MESSAGE'])
    args.add_argument('-m', '--path_to_model', type=str, required=True, help=HELP_MESSAGES['MODEL_MESSAGE'])
    args.add_argument('-c', '--path_to_cldnn_config', type=str, required=False,
                      help=HELP_MESSAGES['CUSTOM_GPU_LIBRARY_MESSAGE'])
    args.add_argument('-l', '--path_to_extension', type=str, required=False, default=None,
                      help=HELP_MESSAGES['CUSTOM_GPU_LIBRARY_MESSAGE'])
    args.add_argument('-api', '--api_type', type=str, required=False, default='async', choices=['sync', 'async'],
                      help=HELP_MESSAGES['API_MESSAGE'])
    args.add_argument('-d', '--target_device', type=str, required=False, default="CPU",
                      help=HELP_MESSAGES['TARGET_DEVICE_MESSAGE'])
    args.add_argument('-niter', '--number_iterations', type=int, required=False, default=None,
                      help=HELP_MESSAGES['ITERATIONS_COUNT_MESSAGE'])
    args.add_argument('-nireq', '--number_infer_requests', type=int, required=False, default=2,
                      help=HELP_MESSAGES['INFER_REQUESTS_COUNT_MESSAGE'])
    args.add_argument('-nthreads', '--number_threads', type=int, required=False, default=None,
                      help=HELP_MESSAGES['INFER_NUM_THREADS_MESSAGE'])
    args.add_argument('-b', '--batch_size', type=int, required=False, default=None,
                      help=HELP_MESSAGES['BATCH_SIZE_MESSAGE'])
    args.add_argument('-pin', '--infer_threads_pinning', type=str, required=False, default='YES',
                      choices=['YES', 'NO'], help=HELP_MESSAGES['INFER_THREADS_PINNING_MESSAGE'])
    return parser.parse_args()


def get_images(path_to_images, batch_size):
    images = list()
    if os.path.isfile(path_to_images):
        while len(images) != batch_size:
            images.append(path_to_images)
    else:
        path = os.path.join(path_to_images, '*')
        files = glob(path, recursive=True)
        for file in files:
            file_extension = file.rsplit('.').pop().upper()
            if file_extension in IMAGE_EXTENSIONS:
                images.append(file)
        if len(images) == 0:
            raise Exception("No images found in {}".format(path_to_images))
        if len(images) < batch_size:
            while len(images) != batch_size:
                images.append(choice(images))
    return images


def get_duration_in_secs(target_device):
    duration = 0
    for device in DEVICE_DURATION_IN_SECS:
        if device in target_device:
            duration = max(duration, DEVICE_DURATION_IN_SECS[device])

    if duration == 0:
        duration = DEVICE_DURATION_IN_SECS[UNKNOWN_DEVICE_TYPE]
        logger.warn("Default duration {} seconds for unknown device {} is used".format(duration, target_device))

    return duration


def fill_blob_with_image(images_path, shape):
    images = np.ndarray(shape)
    for item in range(shape[0]):
        image = cv2.imread(images_path[item])

        new_im_size = tuple(shape[2:])
        if image.shape[:-1] != new_im_size:
            logger.warn("Image {} is resize from ({}) to ({})".format(images_path[item], image.shape[:-1], new_im_size))
            image = cv2.resize(image, new_im_size)

        image = image.transpose((2, 0, 1))
        images[item] = image
    return images


def sync_infer_request(exe_network, times, images):
    iteration_start_time = datetime.now()
    exe_network.infer(images)
    current_time = datetime.now()
    times.append((current_time - iteration_start_time).total_seconds())
    return current_time
