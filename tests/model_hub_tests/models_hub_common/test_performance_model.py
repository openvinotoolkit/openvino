# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import gc
import os
import sys
import time
from enum import Enum
import traceback
import pytest

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


class Results:
    def __init__(self):
        self.converted_infer_time = 0.0
        self.converted_model_time_variance = 0.0
        self.read_model_infer_time = 0.0
        self.read_model_infer_time_variance = 0.0
        self.infer_time_ratio = 0.0
        self.error_message = ''
        self.status = None


def wrap_timer(func, args):
    t0 = time.time()
    retval = func(*args)
    t1 = time.time()
    return retval, t1 - t0


class TestModelPerformance:
    infer_timeout = 600
    threshold_ratio = 0.1
    num_heat_runs = 100
    num_measure_runs = 500
    threshold_var = 10.0

    def load_model(self, model_name, model_link):
        raise "load_model is not implemented"

    def get_inputs_info(self, model_obj):
        raise "get_inputs_info is not implemented"

    def prepare_input(self, input_shape, input_type):
        if input_type in [ov.Type.f32, ov.Type.f64]:
            return 2.0 * rng.random(size=input_shape, dtype=type_map[input_type])
        elif input_type in [ov.Type.u8, ov.Type.u16, ov.Type.i8, ov.Type.i16, ov.Type.i32, ov.Type.i64]:
            return rng.integers(0, 5, size=input_shape).astype(type_map[input_type])
        elif input_type in [str]:
            return np.broadcast_to("Some string", input_shape)
        elif input_type in [bool]:
            return rng.integers(0, 2, size=input_shape).astype(type_map[input_type])
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
        infer_step_t0 = time.time()
        # heat run
        for _ in range(0, TestModelPerformance.num_heat_runs):
            ov_model(inputs)
        # measure
        results = []
        for _ in range(0, TestModelPerformance.num_measure_runs):
            t0 = time.time()
            out_data = ov_model(inputs)
            t1 = time.time()
            results.append(t1 - t0)
        mean = np.mean(results)
        var = np.std(results, ddof=1) * 100 / mean
        infer_step_t1 = time.time()
        print('inference measurement done in {} secs'.format(infer_step_t1 - infer_step_t0))
        return mean, var

    def compile_model(self, model, ie_device):
        core = ov.Core()
        return core.compile_model(model, ie_device)

    def _run(self, model_name, model_link, ie_device):
        results = Results()
        results.status = None
        try:
            print("Load the model {} (url: {})".format(model_name, model_link))
            results.status = Status.LOAD_MODEL
            model_obj, timedelta = wrap_timer(self.load_model, (model_name, model_link))
            print('Model {} loaded in {} secs'.format(model_name, timedelta))
            print("Retrieve inputs info")
            results.status = Status.GET_INPUTS_INFO
            inputs_info, timedelta = wrap_timer(self.get_inputs_info, (model_obj,))
            print('Got inputs info in {} secs'.format(timedelta))
            print("Prepare input data")
            results.status = Status.PREPARE_INPUTS
            inputs = self.prepare_inputs(inputs_info)
            print("Convert the model into ov::Model")
            results.status = Status.GET_CONVERTED_MODEL
            converted_model = self.compile_model(self.get_converted_model(model_obj), ie_device)
            print("read the model into ov::Model")
            results.status = Status.GET_READ_MODEL
            read_model = self.compile_model(self.get_read_model(model_obj), ie_device)
            print("Infer the converted model")
            results.status = Status.INFER_CONVERTED_MODEL
            converted_model_time, converted_model_time_variance = self.infer_model(converted_model, inputs)
            print('converted model time infer {}'.format(converted_model_time))
            print('converted model time infer var {}'.format(converted_model_time_variance))
            print("Infer read model")
            results.status = Status.INFER_READ_MODEL
            read_model_time, read_model_time_variance = self.infer_model(read_model, inputs)
            print('read model time infer {}'.format(read_model_time))
            print('read model time infer var {}'.format(read_model_time_variance))

            infer_time_ratio = converted_model_time/read_model_time

            results.converted_infer_time = converted_model_time
            results.converted_model_time_variance = converted_model_time_variance
            results.read_model_infer_time = read_model_time
            results.read_model_infer_time_variance = read_model_time_variance
            results.infer_time_ratio = infer_time_ratio

            if abs(infer_time_ratio - 1) > TestModelPerformance.threshold_ratio:
                if (read_model_time_variance > TestModelPerformance.threshold_var
                        or converted_model_time_variance > TestModelPerformance.threshold_var):
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

    def run(self, model_name, model_link, ie_device):
        self.result = Results()
        t0 = time.time()
        self.result = multiprocessing_run(self._run, [model_name, model_link, ie_device], model_name, self.infer_timeout)
        t1 = time.time()
        print('test running time {}'.format(t1 - t0))
        if self.result.status == Status.OK:
            return
        err_message = "\n{func} running failed: \n{msg}".format(func=model_name, msg=self.result.error_message)
        if self.result.status == Status.LARGE_INFER_TIME_DIFF_WITH_LARGE_VAR:
            pytest.xfail(err_message)
        else:
            pytest.fail(err_message)
