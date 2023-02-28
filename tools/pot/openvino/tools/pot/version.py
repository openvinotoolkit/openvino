# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from .utils.logger import get_logger

INVALID_VERSION = "invalid version"
logger = get_logger(__name__)


def get_version():
    version_txt = os.path.join(os.path.dirname(os.path.realpath(__file__)), "version.txt")
    if os.path.isfile(version_txt):
        with open(version_txt) as f:
            version = f.readline().replace('\n', '')
        return version

    logger.warning('POT is not installed correctly. Please follow README.md')
    return INVALID_VERSION
