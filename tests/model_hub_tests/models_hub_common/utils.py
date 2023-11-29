# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import shutil

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
