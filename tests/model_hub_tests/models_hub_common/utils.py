# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import shutil
import time

import numpy as np
from models_hub_common.constants import test_device


def get_models_list(file_name: str):
    models = []
    with open(file_name) as f:
        for model_info in f:
            model_info = model_info.strip()
            # skip comment in model scope file
            if model_info.startswith('#'):
                continue
            mark = None
            reason = None
            assert len(model_info.split(',')) == 2 or len(model_info.split(',')) == 4, \
                "Incorrect model info `{}`. It must contain either 2 or 4 fields.".format(model_info)
            if len(model_info.split(',')) == 2:
                model_name, model_link = model_info.split(',')
            elif len(model_info.split(',')) == 4:
                model_name, model_link, mark, reason = model_info.split(',')
                assert mark in ["skip", "xfail"], "Incorrect failure mark for model info {}".format(model_info)
            models.append((model_name, model_link, mark, reason))

    return models


def compare_two_tensors(ov_res, fw_res, eps):
    is_ok = True
    if not np.allclose(ov_res, fw_res, atol=eps, rtol=eps, equal_nan=True):
        is_ok = False
        max_diff = np.abs(ov_res.astype(np.float32) - fw_res.astype(np.float32)).max()
        print("Max diff is {}".format(max_diff))
    else:
        print("Accuracy validation successful!\n")
        print("absolute eps: {}, relative eps: {}".format(eps, eps))
    return is_ok


def get_params(ie_device=None):
    ie_device_params = ie_device if ie_device else test_device

    test_args = []
    for element in itertools.product(ie_device_params):
        test_args.append(element)
    return test_args


def cleanup_dir(dir: str):
    if not os.path.exists(dir):
        return
    # remove all downloaded files from cache
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            pass


def round_num(n: float) -> str:
    if 0.1 < n < 1:
        return str(n)[:4]
    s = '{:.2E}'.format(n)
    if s.endswith('E+00'):
        return s[:-4]
    return s


def nano_secs(secs):
    return float(secs) * (10 ** 9)


def measure(max_time_nano_secs: float, func, args):
    left_time_ns = float(max_time_nano_secs)
    time_slices = []
    n_repeats = 0
    while left_time_ns > 0:
        t0 = time.perf_counter_ns()
        func(*args)
        t1 = time.perf_counter_ns()
        timedelta = t1 - t0
        time_slices.append(timedelta)
        left_time_ns -= timedelta
        n_repeats += 1
    real_runtime_nano_secs = max_time_nano_secs - left_time_ns
    return time_slices, n_repeats, real_runtime_nano_secs


def call_with_timer(timer_label: str, func, args):
    print('{} ...'.format(timer_label))
    t0 = time.time()
    ret_value = func(*args)
    t1 = time.time()
    print('{} is done in {} secs'.format(timer_label, round_num(t1 - t0)))
    return ret_value


def print_stat(s: str, value: float):
    print(s.format(round_num(value)))
