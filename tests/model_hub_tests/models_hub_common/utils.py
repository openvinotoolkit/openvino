# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
import itertools
import os
import shutil
import time

import numpy as np
from models_hub_common.constants import test_device


def parse_list_file(file_name: str):
    with open(file_name, 'r') as f_in:
        for model_info in f_in:
            if not model_info:
                continue
            model_info = model_info.strip()
            # skip comment in model scope file
            if not model_info or model_info.startswith('#'):
                continue
            yield model_info.split(',')


def get_models_list(file_name: str):
    models = []
    for line_items in parse_list_file(file_name):
        if len(line_items) == 2:
            model_name, model_link = line_items
            models.append((model_name, model_link, None, None))
        elif len(line_items) == 4:
            model_name, model_link, mark, reason = line_items
            models.append((model_name, model_link, mark, reason))
        elif len(line_items) > 4:
            model_name, model_link, mark, reason = line_items[:4]
            if not mark:
                mark = None
            if not reason:
                reason = None
            other = line_items[4:]
            transformations = [item[8:] for item in other if item.startswith('ts_name:')]
            layers = [item[6:] for item in other if item.startswith('layer:')]
            models.append((model_name, model_link, mark, reason, transformations, layers))
        else:
            items = ','.join(line_items)
            assert False, \
                f'Incorrect model info fields {items}. It must contain either 2 or 4 or more than 4 fields.'
    return models

def get_skipped_model_links(file_name: str):
    return {line_items[1] for line_items in parse_list_file(file_name)}


def get_models_list_not_skipped(model_list_file: str, skip_list_file: str):
    skipped_links = get_skipped_model_links(skip_list_file)
    not_skipped_models = []
    for model_name, model_link, mark, reason in get_models_list(model_list_file):
        if model_link in skipped_links:
            continue
        not_skipped_models.append((model_name, model_link, mark, reason))
    return not_skipped_models


def compare_two_tensors(ov_res, fw_res, eps):
    is_ok = True
    if ov_res.dtype.type == str or ov_res.dtype.type == np.str_ or ov_res.dtype.type == np.object_:
        ov_res = ov_res.astype('U')
        # TF can represent string tensors in different format: array of bytestreams
        # so we have to align formats of both string tensors, for example, to unicode
        if ov_res.dtype.type != fw_res.dtype.type:
            try:
                fw_res = fw_res.astype('U')
            except:
                # ref_array of object type and each element must be utf-8 decoded
                utf8_decoded_elems = [elem.decode('UTF-8') for elem in fw_res.flatten()]
                fw_res = np.array(utf8_decoded_elems, dtype=str).reshape(fw_res.shape)
        is_ok = np.array_equal(ov_res, fw_res)
    elif ov_res.dtype == bool:
        is_ok = np.array_equal(ov_res, fw_res)
    elif not np.allclose(ov_res, fw_res, atol=eps, rtol=eps, equal_nan=True):
        is_ok = False
        max_diff = np.abs(ov_res.astype(np.float32) - fw_res.astype(np.float32)).max()
        print("Max diff is {}".format(max_diff))

    if is_ok:
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


def retry(max_retries=3, exceptions=(Exception,), delay=None):
    def retry_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f"Attempt {attempt + 1} of {max_retries} failed: {e}")
                    if attempt < max_retries - 1 and delay is not None:
                        time.sleep(delay)
                    else:
                        raise e
        return wrapper
    return retry_decorator
