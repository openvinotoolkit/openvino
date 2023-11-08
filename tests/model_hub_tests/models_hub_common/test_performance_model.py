# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import gc
import os

import numpy as np
from models_hub_common.multiprocessing_utils import multiprocessing_run
import openvino as ov

import tensorflow_text  # do not delete, needed for text models

# set seed to have deterministic input data generation
# to avoid sporadic issues in inference results
rng = np.random.default_rng(seed=56190)

type_map = {
    ov.Type.f64: np.float64,
    ov.Type.f32: np.float32,
    ov.Type.i8: np.int8,
    ov.Type.i16: np.int16,
    ov.Type.i32: np.int32,
    ov.Type.i64: np.int64,
    ov.Type.u8: np.uint8,
    ov.Type.u16: np.uint16,
    ov.Type.boolean: bool,
}

class TestPerformanceModel:
    infer_timeout = 600

    def load_model(self, model_name, model_link):
        raise "load_model is not implemented"

    def get_inputs_info(self, model_obj):
        raise "get_inputs_info is not implemented"

    def prepare_input(self, input_shape, input_type):
        if input_type in [ov.Type.f32, ov.Type.f64]:
            return 2.0 * rng.random(size=input_shape, dtype=type_map[input_type])
        elif input_type in [ov.Type.u8, ov.Type.u16, ov.Type.i8, ov.Type.i16, ov.Type.i32, ov.Type.i64]:
            return rng.integers(0, 5, size=input_shape).astype(input_type)
        elif input_type in [str]:
            return np.broadcast_to("Some string", input_shape)
        elif input_type in [bool]:
            return rng.integers(0, 2, size=input_shape).astype(input_type)
        else:
            assert False, "Unsupported type {}".format(input_type)

    def prepare_inputs(self, inputs_info):
        if len(inputs_info) > 0 and inputs_info[0] == 'list':
            inputs = []
            inputs_info = inputs_info[1:]
            for input_name, input_shape, input_type in inputs_info:
                inputs.append(self.prepare_input(input_shape, input_type))
        else:
            inputs = {}
            for input_name, input_shape, input_type in inputs_info:
                inputs[input_name] = self.prepare_input(input_shape, input_type)
        return inputs

    def get_converted_model(self, model_path: str):
        raise "get_converted_model is not implemented"

    def get_read_model(self, model_path: str):
        raise "get_read_model is not implemented"

    def infer_model(self, ov_model, inputs):
        ov_model(inputs)

    def compile_model(self, model, ie_device):
        core = ov.Core()
        return core.compile_model(model, ie_device)

    def _run(self, model_name, model_link, ie_device):
        print("Load the model {} (url: {})".format(model_name, model_link))
        model_path = self.load_model(model_name, model_link)
        print("Retrieve inputs info")
        inputs_info = self.get_inputs_info(model_path)
        print("Prepare input data")
        inputs = self.prepare_inputs(inputs_info)
        print("Convert the model into ov::Model")
        converted_model = self.compile_model(self.get_converted_model(model_path), ie_device)
        print("read the model into ov::Model")
        read_model = self.compile_model(self.get_read_model(model_path), ie_device)
        print("Infer the converted model")
        self.infer_model(converted_model, inputs)
        print("Infer read model")
        self.infer_model(read_model, inputs)

    def run(self, model_name, model_link, ie_device):
        multiprocessing_run(self._run, [model_name, model_link, ie_device], model_name, self.infer_timeout)
