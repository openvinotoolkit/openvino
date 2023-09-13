# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np

from openvino.runtime import Core
from openvino.tools.mo.convert import convert_model


def basic_check(input_model, argv_input, input_data, expected_dtype, expected_value, freeze_placeholder_with_value=None,
                input_shape=None, only_conversion=False, input_model_is_text=True, use_new_frontend=True,
                use_legacy_frontend=False, extensions=None, input_checkpoint=None):
    path = os.path.dirname(__file__)
    input_model = os.path.join(path, "test_models", input_model)

    ov_model = convert_model(input_model, input=argv_input,
                             freeze_placeholder_with_value=freeze_placeholder_with_value,
                             input_shape=input_shape, input_model_is_text=input_model_is_text,
                             use_new_frontend=use_new_frontend, use_legacy_frontend=use_legacy_frontend,
                             framework="tf", extensions=extensions, input_checkpoint=input_checkpoint)

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
