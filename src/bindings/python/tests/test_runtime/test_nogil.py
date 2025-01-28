# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import os
import io
import threading
import numpy as np
import openvino.properties as props

from openvino import Core, Model, AsyncInferQueue, PartialShape, Layout, serialize
from openvino import opset13 as ops
from openvino.preprocess import PrePostProcessor

from tests import skip_devtest


# check if func releases the GIL and doesn't increment reference counters of args while GIL is released
def check_gil_released_safe(func, is_assign=False, args=[]):  # noqa: B006
    global gil_released
    gil_released = False

    def detect_gil():
        global gil_released
        # while sleeping main thread acquires GIL and runs func, which will release GIL
        time.sleep(0.000001)
        # increment reference counting of args while running func
        args_ = args  # noqa: F841 'assigned to but never used'
        gil_released = True
        time.sleep(0.1)

    thread = threading.Thread(target=detect_gil)
    thread.start()

    if is_assign:
        _ = func(*args)
    else:
        func(*args)

    count = threading.active_count()

    assert count == 2

    if not gil_released:
        thread.join()
        pytest.xfail(reason="Depend on condition race")
    else:
        thread.join()


device = os.environ.get("TEST_DEVICE") if os.environ.get("TEST_DEVICE") else "CPU"
core = Core()
core.set_property({props.enable_profiling: True})
param = ops.parameter([224, 224])
model = Model(ops.relu(param), [param])
compiled_model = core.compile_model(model, device)
infer_queue = AsyncInferQueue(compiled_model, 1)
user_stream = io.BytesIO()


# AsyncInferQueue

@skip_devtest
def test_gil_released_async_infer_queue_start_async():
    infer_queue.start_async()
    check_gil_released_safe(infer_queue.start_async, False)
    infer_queue.wait_all()


@skip_devtest
def test_gil_released_async_infer_queue_is_ready():
    infer_queue.start_async()
    check_gil_released_safe(infer_queue.is_ready, False)
    infer_queue.wait_all()


@skip_devtest
def test_gil_released_async_infer_queue_wait_all():
    infer_queue.start_async()
    check_gil_released_safe(infer_queue.wait_all, False)
    infer_queue.wait_all()


@skip_devtest
def test_gil_released_async_infer_queue_get_idle_request_id():
    infer_queue.start_async()
    check_gil_released_safe(infer_queue.get_idle_request_id, True)
    infer_queue.wait_all()


# CompiledModel

@skip_devtest
def test_gil_released_create_infer_request():
    check_gil_released_safe(compiled_model.create_infer_request, True)


@skip_devtest
def test_gil_released_infer_new_request():
    check_gil_released_safe(compiled_model, True)


@skip_devtest
def test_gil_released_export():
    check_gil_released_safe(compiled_model.export_model, True)


@skip_devtest
def test_gil_released_export_advanced():
    check_gil_released_safe(compiled_model.export_model, False, [user_stream])
    check_gil_released_safe(core.import_model, True, [user_stream, device])


@skip_devtest
def test_gil_released_get_runtime_model():
    check_gil_released_safe(compiled_model.get_runtime_model, True)


# Core

@skip_devtest
def test_compile_model(device):
    check_gil_released_safe(core.compile_model, True, [model, device])


@skip_devtest
def test_read_model_from_bytes():
    bytes_model = bytes(b"""<net name="relu_model" version="11">
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
    check_gil_released_safe(core.read_model, True, [bytes_model])


@skip_devtest
def test_read_model_from_path():
    from pathlib import Path
    model_path = "relu.xml"
    bin_path = "relu.bin"
    serialize(model, model_path, bin_path)
    check_gil_released_safe(core.read_model, True, [Path(model_path)])
    os.remove(model_path)
    os.remove(bin_path)


@skip_devtest
def test_query_model(device):
    check_gil_released_safe(core.query_model, True, [model, device])


@skip_devtest
def test_get_available_devices(device):
    check_gil_released_safe(getattr, True, [core, "available_devices"])


# InferRequest

request = compiled_model.create_infer_request()


@skip_devtest
def test_infer_assign():
    data = [np.random.normal(size=list(compiled_model.input().shape))]
    check_gil_released_safe(request.infer, True, [data])


@skip_devtest
def test_infer_no_assign():
    data = [np.random.normal(size=list(compiled_model.input().shape))]
    check_gil_released_safe(request.infer, False, [data])


@skip_devtest
def test_start_async():
    data = [np.random.normal(size=list(compiled_model.input().shape))]
    check_gil_released_safe(request.start_async, False, [data])
    request.wait()


@skip_devtest
def test_wait():
    data = [np.random.normal(size=list(compiled_model.input().shape))]
    request.start_async(data)
    check_gil_released_safe(request.wait, False)
    request.wait()


@skip_devtest
def test_wait_for():
    data = [np.random.normal(size=list(compiled_model.input().shape))]
    request.start_async(data)
    check_gil_released_safe(request.wait_for, False, [1])
    request.wait()


@skip_devtest
def test_get_profiling_info():
    check_gil_released_safe(request.get_profiling_info, True)


@skip_devtest
def test_query_state():
    check_gil_released_safe(request.query_state, True)


# Preprocessing

@skip_devtest
def test_pre_post_process_build():
    ppp = PrePostProcessor(model)
    ppp.input().model().set_layout(Layout("NC"))
    check_gil_released_safe(ppp.build, True)


@skip_devtest
def test_model_reshape():
    check_gil_released_safe(model.reshape, False, [PartialShape([128, 128])])
    check_gil_released_safe(model.reshape, False, [[164, 164]])
    check_gil_released_safe(model.reshape, False, [(178, 178)])
    check_gil_released_safe(model.reshape, False, ["194, 194"])
    check_gil_released_safe(model.reshape, False, [{0: [224, 224]}])
