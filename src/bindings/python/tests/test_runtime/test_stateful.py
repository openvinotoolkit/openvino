# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pytest

from openvino import (
    Core,
    CompiledModel,
    InferRequest,
    Tensor,
    compile_model,
)

from tests.utils.helpers import generate_model_with_memory
from openvino.utils.types import get_dtype


class Caller:
    def __init__(self, base):
        self.base = base

    def query_state(self):
        return self.base.query_state()

    def reset_state(self):
        return self.base.reset_state()

    def call(self, *args, **kwargs):
        if type(self.base) is CompiledModel:
            return self.base(*args, **kwargs)
        elif type(self.base) is InferRequest:
            return self.base.infer(*args, **kwargs)
        else:
            raise RuntimeError("Unknown caller base!")


@pytest.mark.parametrize("base_class",
                         [CompiledModel,
                          InferRequest])
@pytest.mark.parametrize("data_type",
                         [np.float32,
                          np.int32,
                          np.float16])
@pytest.mark.parametrize("mode", ["set_init_memory_state", "reset_memory_state", "normal", "reset_via_infer_request"])
@pytest.mark.parametrize("input_shape", [[10], [10, 10], [10, 10, 10], [2, 10, 10, 10]])
@pytest.mark.skipif(
    os.environ.get("TEST_DEVICE", "CPU") not in ["CPU", "GPU"],
    reason=f"Can't run test on device {os.environ.get('TEST_DEVICE', 'CPU')}, "
    "Memory layers fully supported only on CPU and GPU",
)
def test_query_state_write_buffer(device, base_class, input_shape, data_type, mode):
    core = Core()

    model = generate_model_with_memory(input_shape, data_type)
    model.validate_nodes_and_infer_types()
    compiled_model = core.compile_model(model=model, device_name=device)
    if base_class is CompiledModel:
        caller = Caller(base=compiled_model)
    else:  # InferRequest
        caller = Caller(base=compiled_model.create_infer_request())
    mem_states = caller.query_state()
    mem_state = mem_states[0]

    assert mem_state.name == "var_id_667"
    assert get_dtype(mem_state.state.element_type) == data_type

    for i in range(1, 10):
        if mode == "set_init_memory_state":
            # create initial value
            const_init = 5
            init_array = np.full(input_shape, const_init, dtype=get_dtype(mem_state.state.element_type))
            tensor = Tensor(init_array)
            mem_state.state = tensor

            res = caller.call({0: np.full(input_shape, 1, dtype=data_type)})
            expected_res = np.full(input_shape, 1 + const_init, dtype=data_type)
        elif mode == "reset_memory_state":
            # reset initial state of ReadValue to zero
            mem_state.reset()
            res = caller.call({0: np.full(input_shape, 1, dtype=data_type)})
            # always ones
            expected_res = np.full(input_shape, 1, dtype=data_type)
        elif mode == "reset_via_infer_request":
            # reset initial state of ReadValue to zero
            caller.reset_state()
            res = caller.call({0: np.full(input_shape, 1, dtype=data_type)})
            # always ones
            expected_res = np.full(input_shape, 1, dtype=data_type)
        else:
            res = caller.call({0: np.full(input_shape, 1, dtype=data_type)})
            expected_res = np.full(input_shape, i, dtype=data_type)

        assert np.allclose(res[list(res)[0]], expected_res, atol=1e-6), f"Expected values: {expected_res} \n Actual values: {res} \n"
