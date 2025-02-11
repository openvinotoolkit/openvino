# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from sys import stdout
from os import environ

from . import constants

def get_logger(app_name: str):
    logging.basicConfig()
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.INFO)
    return logger

UTILS_LOGGER = get_logger('conformance_utilities')


def progressbar(it_num, message="", progress_bar_size=60, out=stdout):
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


def set_env_variable(env: environ, var_name: str, var_value: str):
    if var_name in env and not var_value in env[var_name]:
        env[var_name] = var_value + constants.ENV_SEPARATOR + env[var_name]
    else:
        env[var_name] = var_value
    return env
