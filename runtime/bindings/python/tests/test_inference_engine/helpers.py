# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import cv2
import os


def image_path():
    path_to_repo = os.environ["DATA_PATH"]
    path_to_img = os.path.join(path_to_repo, "validation_set", "224x224", "dog.bmp")
    return path_to_img


def model_path(is_myriad=False):
    path_to_repo = os.environ["MODELS_PATH"]
    if not is_myriad:
        test_xml = os.path.join(path_to_repo, "models", "test_model", "test_model_fp32.xml")
        test_bin = os.path.join(path_to_repo, "models", "test_model", "test_model_fp32.bin")
    else:
        test_xml = os.path.join(path_to_repo, "models", "test_model", "test_model_fp16.xml")
        test_bin = os.path.join(path_to_repo, "models", "test_model", "test_model_fp16.bin")
    return (test_xml, test_bin)


def read_image():
    n, c, h, w = (1, 3, 32, 32)
    image = cv2.imread(image_path())
    if image is None:
        raise FileNotFoundError("Input image not found")

    image = cv2.resize(image, (h, w)) / 255
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image = image.reshape((n, c, h, w))
    return image
