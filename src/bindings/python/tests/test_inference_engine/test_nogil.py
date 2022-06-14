# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import os
import io
from threading import Thread
import numpy as np

from openvino.runtime import Core, Model, AsyncInferQueue, PartialShape, Layout, opset8 as ops, serialize
from openvino.preprocess import PrePostProcessor


# check if func releases the GIL and doens't increment reference counters of args while GIL is released
def check_gil_released_safe(func, args=[]):  # noqa: B006
    global gil_released
    gil_released = False

    def detect_gil():
        global gil_released
        # while sleeping main thread acquires GIL and runs func, which will release GIL
        time.sleep(0.000001)
        # increment reference counting of args while running func
        args_ = args  # noqa: F841 'assigned to but never used'
        gil_released = True
    thread = Thread(target=detect_gil)
    thread.start()
    func(*args)
    if not gil_released:
        pytest.xfail(reason="Depend on condition race")
    thread.join()


device = os.environ.get("TEST_DEVICE") if os.environ.get("TEST_DEVICE") else "CPU"
core = Core()
core.set_property({"PERF_COUNT": "YES"})
param = ops.parameter([224, 224])
model = Model(ops.relu(param), [param])
compiled = core.compile_model(model, device)
infer_queue = AsyncInferQueue(compiled, 1)
user_stream = io.BytesIO()


# AsyncInferQueue

def test_gil_released_async_infer_queue_start_async():
    infer_queue.start_async()
    check_gil_released_safe(infer_queue.start_async)


def test_gil_released_async_infer_queue_is_ready():
    infer_queue.start_async()
    check_gil_released_safe(infer_queue.is_ready)


def test_gil_released_async_infer_queue_wait_all():
    infer_queue.start_async()
    check_gil_released_safe(infer_queue.wait_all)


def test_gil_released_async_infer_queue_get_idle_request_id():
    infer_queue.start_async()
    check_gil_released_safe(infer_queue.get_idle_request_id)


# CompiledModel

def test_gil_released_create_infer_request():
    check_gil_released_safe(compiled.create_infer_request)


def test_gil_released_infer_new_request():
    check_gil_released_safe(compiled)


def test_gil_released_export():
    check_gil_released_safe(compiled.export_model)


def test_gil_released_export_advanced():
    check_gil_released_safe(compiled.export_model, [user_stream])


def test_gil_released_get_runtime_model():
    check_gil_released_safe(compiled.get_runtime_model)


# Core

def test_compile_model(device):
    check_gil_released_safe(core.compile_model, [model, device])


def test_read_model_from_bytes():
    ir = bytes(b"""<net name="relu_model" version="11">
    <layers>
        <layer id="0" name="x" type="Parameter" version="opset1">
            <data element_type="f32" shape="10"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="relu" type="ReLU" version="opset1">
            <input>
                <port id="0">
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="result" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>10</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
    </edges>
</net>""")
    check_gil_released_safe(core.read_model, [ir])


def test_read_model_from_path():
    from pathlib import Path
    model_path = "relu.xml"
    bin_path = "relu.bin"
    serialize(model, model_path, bin_path)
    check_gil_released_safe(core.read_model, [Path(model_path)])
    os.remove(model_path)
    os.remove(bin_path)


def test_import_model(device):
    check_gil_released_safe(core.import_model, [user_stream, device])


def test_query_model(device):
    check_gil_released_safe(core.query_model, [model, device])


def test_get_available_devices(device):
    check_gil_released_safe(getattr, [core, "available_devices"])


# InferRequest

request = compiled.create_infer_request()


def test_infer():
    data = [np.random.normal(size=list(compiled.input().shape))]
    check_gil_released_safe(request.infer, [data])


def test_start_async():
    data = [np.random.normal(size=list(compiled.input().shape))]
    check_gil_released_safe(request.start_async, [data])
    request.wait()


def test_wait():
    data = [np.random.normal(size=list(compiled.input().shape))]
    request.start_async(data)
    check_gil_released_safe(request.wait)


def test_wait_for():
    data = [np.random.normal(size=list(compiled.input().shape))]
    request.start_async(data)
    check_gil_released_safe(request.wait_for, [1])
    request.wait()


def test_get_profiling_info():
    check_gil_released_safe(request.get_profiling_info)


def test_query_state():
    check_gil_released_safe(request.query_state)


# Preprocessing

def test_pre_post_process_build():
    ppp = PrePostProcessor(model)
    ppp.input().model().set_layout(Layout("NC"))
    check_gil_released_safe(ppp.build)


def test_model_reshape():
    check_gil_released_safe(model.reshape, [PartialShape([128, 128])])
    check_gil_released_safe(model.reshape, [[164, 164]])
    check_gil_released_safe(model.reshape, [(178, 178)])
    check_gil_released_safe(model.reshape, ["194, 194"])
    check_gil_released_safe(model.reshape, [{0: [224, 224]}])
