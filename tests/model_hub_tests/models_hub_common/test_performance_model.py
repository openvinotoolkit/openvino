# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import gc
import os

import numpy as np
from models_hub_common.multiprocessing_utils import multiprocessing_run
from openvino import convert_model
from openvino.runtime import Core

import tensorflow_text  # do not delete, needed for text models

# set seed to have deterministic input data generation
# to avoid sporadic issues in inference results
rng = np.random.default_rng(seed=56190)


class TestPerformanceModel:
    infer_timeout = 600

    def load_model(self, model_name, model_link):
        raise "load_model is not implemented"

    def get_inputs_info(self, model_obj):
        raise "get_inputs_info is not implemented"

    def prepare_input(self, input_shape, input_type):
        if input_type in [np.float32, np.float64]:
            return 2.0 * rng.random(size=input_shape, dtype=input_type)
        elif input_type in [np.uint8, np.uint16, np.int8, np.int16, np.int32, np.int64]:
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

    def convert_model(self, model_path: str):
        ov_model = convert_model(model_path)
        return ov_model

    def infer_fw_model(self, model_obj, inputs):
        raise "infer_fw_model is not implemented"

    def infer_ov_model(self, ov_model, inputs, ie_device):
        core = Core()
        compiled = core.compile_model(ov_model, ie_device)
        ov_outputs = compiled(inputs)
        return ov_outputs

    def _run(self, model_name, model_link, ie_device):
        print("Load the model {} (url: {})".format(model_name, model_link))
        fw_model = self.load_model(model_name, model_link)
        print("Retrieve inputs info")
        inputs_info = self.get_inputs_info(fw_model)
        print("Prepare input data")
        inputs = self.prepare_inputs(inputs_info)
        print("Convert the model into ov::Model")
        ov_model = self.convert_model(fw_model)
        print("Infer the original model")
        fw_outputs = self.infer_fw_model(fw_model, inputs)
        print("Infer ov::Model")
        ov_outputs = self.infer_ov_model(ov_model, inputs, ie_device)
        # TODO: run series

    def run(self, model_name, model_link, ie_device):
        multiprocessing_run(self._run, [model_name, model_link, ie_device], model_name, self.infer_timeout)
