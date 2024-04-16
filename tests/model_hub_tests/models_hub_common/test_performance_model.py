# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import sys
import traceback
import time
from enum import Enum
import pytest
from openvino.runtime.utils.types import openvino_to_numpy_types_map
import models_hub_common.utils as utils
import models_hub_common.constants as const


import numpy as np
import openvino as ov
import pytest
from models_hub_common.multiprocessing_utils import multiprocessing_run
from openvino.runtime.utils.types import openvino_to_numpy_types_map

# set seed to have deterministic input data generation
# to avoid sporadic issues in inference results
rng = np.random.default_rng(seed=56190)


def get_numpy_type(ov_type):
    np_type = next(
        (np_type_value for (ov_type_value, np_type_value) in openvino_to_numpy_types_map if ov_type_value == ov_type),
        None,
    )

    if not np_type:
        raise Exception('no numpy type for type {} found'.format(ov_type))

    return np_type


class Status(Enum):
    OK = 0
    LARGE_INFER_TIME_DIFF = 1
    LOAD_MODEL = 2
    GET_INPUTS_INFO = 3
    PREPARE_INPUTS = 4
    GET_CONVERTED_MODEL = 5
    GET_READ_MODEL = 6
    INFER_CONVERTED_MODEL = 7
    INFER_READ_MODEL = 8
    LARGE_INFER_TIME_DIFF_WITH_LARGE_VAR = 9


class ModelResults:
    def __init__(self):
        self.infer_mean_time = 0.0
        self.infer_variance = 0.0


class Results:
    def __init__(self):
        self.converted_model_results = ModelResults()
        self.read_model_results = ModelResults()
        self.infer_time_ratio = 0.0
        self.error_message = ''
        self.status = None
        self.model_name = ''
        self.model_link = ''
        self.ie_device = ''


class TestModelPerformance:
    infer_timeout = 600
    threshold_ratio = 0.1
    threshold_var = 10.0

    def load_model(self, model_name, model_link):
        raise "load_model is not implemented"

    def prepare_input(self, input_shape, input_type):
        if input_type in [ov.Type.f32, ov.Type.f64]:
            return 2.0 * rng.random(size=input_shape, dtype=get_numpy_type(input_type))
        elif input_type in [ov.Type.u8, ov.Type.u16, ov.Type.i8, ov.Type.i16, ov.Type.i32, ov.Type.i64]:
            return rng.integers(0, 5, size=input_shape).astype(get_numpy_type(input_type))
        elif input_type in [str]:
            return np.broadcast_to("Some string", input_shape)
        elif input_type in [bool]:
            return rng.integers(0, 2, size=input_shape).astype(get_numpy_type(input_type))
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

    def get_inputs_info(self, model_path: str):
        inputs_info = []
        core = ov.Core()
        model = core.read_model(model=model_path)
        for param in model.inputs:
            input_shape = []
            param_shape = param.get_node().get_output_partial_shape(0)
            shape_special_dims = [ov.Dimension(), ov.Dimension(), ov.Dimension(), ov.Dimension(3)]
            if param_shape == ov.PartialShape(shape_special_dims) and param.get_element_type() == ov.Type.f32:
                # image classification case, let us imitate an image
                # that helps to avoid compute output size issue
                input_shape = [1, 200, 200, 3]
            else:
                for dim in param_shape:
                    if dim.is_dynamic:
                        input_shape.append(1)
                    else:
                        input_shape.append(dim.get_length())
            inputs_info.append((param.get_node().get_friendly_name(), input_shape, param.get_element_type()))
        return inputs_info

    def get_read_model(self, model_path: str):
        core = ov.Core()
        return core.read_model(model=model_path)

    def heat_hardware(self, ov_model, inputs, conf) -> None:
        _, heat_n_repeats, _ = utils.measure(conf.runtime_heat_duration, ov_model, (inputs,))
        print('heat done in {} repeats'.format(heat_n_repeats))

    def measure_inference(self, ov_model, inputs, conf) -> ModelResults:
        time_slices, infer_n_repeats, real_runtime = utils.measure(conf.runtime_measure_duration, ov_model, (inputs,))
        print('measurement done in {} repeats'.format(infer_n_repeats))
        infer_throughput = float(infer_n_repeats * (10 ** 9)) / real_runtime
        infer_mean_time_ns = np.mean(time_slices)
        infer_mean_time = infer_mean_time_ns / (10 ** 9)
        infer_variance = (np.std(time_slices, ddof=1) * 100) / infer_mean_time_ns
        utils.print_stat('model time infer {} secs', infer_mean_time)
        utils.print_stat('model time infer var {}', infer_variance)
        utils.print_stat('model time infer throughput {}', infer_throughput)
        results = ModelResults()
        results.infer_mean_time = infer_mean_time
        results.infer_variance = infer_variance
        return results

    def infer_model(self, ov_model, inputs, conf) -> ModelResults:
        self.heat_hardware(ov_model, inputs, conf)
        return self.measure_inference(ov_model, inputs, conf)

    def compile_model(self, model, ie_device):
        core = ov.Core()
        return core.compile_model(model, ie_device)

    def __run(self, model_name, model_link, ie_device, conf):
        results = Results()
        results.model_name = model_name
        results.model_link = model_link
        results.ie_device = ie_device
        results.status = None
        try:
            results.status = Status.LOAD_MODEL
            model_obj = utils.call_with_timer('Load model', self.load_model, (model_name, model_link))
            results.status = Status.GET_INPUTS_INFO
            inputs_info = utils.call_with_timer('Retrieve model inputs', self.get_inputs_info, (model_obj,))
            results.status = Status.PREPARE_INPUTS
            inputs = self.prepare_inputs(inputs_info)
            results.status = Status.GET_CONVERTED_MODEL
            model = utils.call_with_timer('Convert model', ov.convert_model, (model_obj,))
            converted_model = utils.call_with_timer('Compile converted model', self.compile_model, (model, ie_device))
            results.status = Status.GET_READ_MODEL
            model = utils.call_with_timer('Read model', self.get_read_model, (model_obj,))
            read_model = utils.call_with_timer('Compile read model', self.compile_model, (model, ie_device))
            results.status = Status.INFER_CONVERTED_MODEL
            results.converted_model_results = utils.call_with_timer('Infer converted model',
                                                                    self.infer_model,
                                                                    (converted_model, inputs, conf))
            results.status = Status.INFER_READ_MODEL
            results.read_model_results = utils.call_with_timer('Infer read model',
                                                               self.infer_model,
                                                               (read_model, inputs, conf))

            infer_time_ratio = (results.converted_model_results.infer_mean_time /
                                results.read_model_results.infer_mean_time)
            utils.print_stat('infer ratio converted_model_time/read_model_time {}', infer_time_ratio)

            results.infer_time_ratio = infer_time_ratio

            if abs(infer_time_ratio - 1) > TestModelPerformance.threshold_ratio:
                if (results.read_model_results.infer_variance > TestModelPerformance.threshold_var
                        or results.converted_model_results.infer_variance > TestModelPerformance.threshold_var):
                    results.status = Status.LARGE_INFER_TIME_DIFF_WITH_LARGE_VAR
                    results.error_message = "too large ratio {} with large variance".format(infer_time_ratio)
                else:
                    results.status = Status.LARGE_INFER_TIME_DIFF
                    results.error_message = "too large ratio {}".format(infer_time_ratio)
            else:
                results.status = Status.OK
        except:
            ex_type, ex_value, tb = sys.exc_info()
            results.error_message = "{tb}\n{ex_type}: {ex_value}".format(tb=''.join(traceback.format_tb(tb)),
                                                                         ex_type=ex_type.__name__, ex_value=ex_value)
        return results

    def run(self, model_name, model_link, ie_device, conf):
        self.result = Results()
        t0 = time.time()
        self.result = multiprocessing_run(self.__run, [model_name, model_link, ie_device, conf], model_name,
                                          self.infer_timeout)
        t1 = time.time()
        utils.print_stat('test run time {} secs', (t1 - t0))
        if self.result.status == Status.OK:
            return
        err_message = "\n{func} running failed: \n{msg}".format(func=model_name, msg=self.result.error_message)
        if self.result.status == Status.LARGE_INFER_TIME_DIFF_WITH_LARGE_VAR:
            pytest.xfail(err_message)
        else:
            pytest.fail(err_message)
