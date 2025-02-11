# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np

from openvino import Core
from openvino.tools.ovc.convert import convert_model


def basic_check(input_model, argv_input, input_data, expected_dtype, expected_value, \
    only_conversion=False, input_model_is_text=True, use_new_frontend=True, extensions=None):
    path = os.path.dirname(__file__)
    if isinstance(input_model, (tuple, list)):
        input_model = tuple(os.path.join(path, "test_models", part) for part in input_model)
    else:
        input_model = os.path.join(path, "test_models", input_model)

    ov_model = convert_model(input_model, input=argv_input, extension=extensions)

    if only_conversion:
        return ov_model

    ie = Core()
    exec_net = ie.compile_model(ov_model, "CPU")
    req = exec_net.create_infer_request()
    results = req.infer(input_data)
    values = list(results.values())[0]
    if expected_dtype is not None:
        assert values.dtype == expected_dtype
    assert np.allclose(values,
                       expected_value), "Expected and actual values are different." \
                                        " Expected value: {}, actual value: {}".format(expected_value, values)

    return ov_model
