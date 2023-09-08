# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import numpy as np

from models_hub_common.constants import test_device


def get_models_list(file_name: str):
    models = []
    with open(file_name) as f:
        for model_info in f:
            # skip comment in model scope file
            if model_info.startswith('#'):
                continue
            mark = None
            reason = None
            assert len(model_info.split(',')) == 2 or len(model_info.split(',')) == 4, \
                "Incorrect model info `{}`. It must contain either 2 or 4 fields.".format(
                    model_info)
            if len(model_info.split(',')) == 2:
                model_name, model_link = model_info.split(',')
            elif len(model_info.split(',')) == 4:
                model_name, model_link, mark, reason = model_info.split(',')
                assert mark == "skip", "Incorrect failure mark for model info {}".format(
                    model_info)
            models.append((model_name, model_link, mark, reason))

    return models


def compare_two_tensors(ov_res, fw_res, eps):
    is_ok = True
    if not np.allclose(ov_res, fw_res, atol=eps, rtol=eps, equal_nan=True):
        is_ok = False
        print("Max diff is {}".format(np.array(abs(ov_res - fw_res)).max()))
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
