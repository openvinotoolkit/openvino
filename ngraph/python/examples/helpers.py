# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper functions."""

import numpy as np
import os


def get_examples_path() -> str:
    """Return examples absoulte path."""
    return os.path.dirname(os.path.realpath(__file__))


def get_example_model_path() -> str:
    """Return example model path to xml file."""
    return get_examples_path() + '/model/test_model_fp32.xml'


def get_example_weights_path() -> str:
    """Return example model path to bin file."""
    return get_examples_path() + '/model/test_model_fp32.bin'


def generate_random_images(num=1) -> np.array:
    """Generate `num` random images."""
    return [np.array(np.random.rand(1, 3, 32, 32), dtype=np.float32)
            for i in range(0, num)]
