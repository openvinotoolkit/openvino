# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from shutil import rmtree, copyfile
from zipfile import ZipFile, is_zipfile

import tarfile

from shutil import rmtree, copyfile
import sys
from pathlib import Path, PurePath

from urllib.parse import urlparse

from . import constants

def get_logger(app_name: str):
    logging.basicConfig()
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.INFO)
    return logger

UTILS_LOGGER = get_logger('conformance_utilities')


def progressbar(it_num, message="", progress_bar_size=60, out=sys.stdout):
    max_len = len(it_num)
    if max_len == 0:
        return
    def show(sym_pos):
        x = int(progress_bar_size * sym_pos / max_len)
        print("{}[{}{}] {}/{}".format(message, "#"*x, "."*(progress_bar_size-x), sym_pos, max_len), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it_num):
        yield item
        show(i+1)
    print("", flush=True, file=out)


def set_env_variable(env: os.environ, var_name: str, var_value: str):
    if var_name in env:
        env[var_name] = var_value + constants.ENV_SEPARATOR + env[var_name]
    else:
        env[var_name] = var_value
    return env
