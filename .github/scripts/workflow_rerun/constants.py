# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os


GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')


def init_logger():
    LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(level=LOGLEVEL,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d-%Y %H:%M:%S')


init_logger()

LOGGER = logging.getLogger('rerunner')
