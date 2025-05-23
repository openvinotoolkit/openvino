# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
import numpy as np
import pytest

import openvino.opset13 as ops
from openvino import Core, CompiledModel, InferRequest, Model
from openvino import ConstOutput
from openvino.utils.data_helpers import OVDict


def _get_ovdict(
    device,
    input_shape=None,
    data_type=np.float32,
    input_names=None,
    output_names=None,
    multi_output=False,
    direct_infer=False,
    split_num=5,
):
    # Create model
    # If model is multi-output (multi_output=True), input_shape must match
    # requirements of split operation.
    # TODO OpenSource: refactor it to be more generic
    if input_shape is None:
        input_shape = [1, 20]
    if input_names is None:
        input_names = ["data_0"]
    if output_names is None:
        output_names = ["output_0"]
    if multi_output:
        assert isinstance(output_names, (list, tuple))
        assert len(output_names) > 1
        assert len(output_names) == split_num
    param = ops.parameter(input_shape, data_type, name=input_names[0])
    model = Model(
        ops.split(param, 1, split_num) if multi_output else ops.abs(param), [param],
    )
    # Manually name outputs
    for i in range(len(output_names)):
        model.output(i).tensor.names = {output_names[i]}
    # Compile model
    core = Core()
    compiled_model = core.compile_model(model, device)
    # Create test data
    input_data = np.random.random(input_shape).astype(data_type)
    # Two ways of infering
    if direct_infer:
        result = compiled_model(input_data)
        assert result is not None
        return result, compiled_model

    request = compiled_model.create_infer_request()
    result = request.infer(input_data)
    assert result is not None
    return result, request


def _check_keys(keys, outs):
    outs_iter = iter(outs)
    for key in keys:
        assert isinstance(key, ConstOutput)
        assert key == next(outs_iter)
    return True


def _check_values(result):
    for value in result.values():
        assert isinstance(value, np.ndarray)
    return True


def _check_items(result, outs, output_names):
    i = 0
    for key, value in result.items():
        assert isinstance(key, ConstOutput)
        assert isinstance(value, np.ndarray)
        # Check values
        assert np.equal(result[outs[i]], result[key]).all()
        assert np.equal(result[outs[i]], result[i]).all()
        assert np.equal(result[outs[i]], result[output_names[i]]).all()
        i += 1
    return True


def _check_dict(result, obj, output_names=None):
    if output_names is None:
        output_names = ["output_0"]

    outs = obj.model_outputs if isinstance(obj, InferRequest) else obj.outputs
    assert len(outs) == len(result)
    assert len(outs) == len(output_names)
    # Check for __iter__
    assert _check_keys(result, outs)
    # Check for keys function
    assert _check_keys(result.keys(), outs)
    assert _check_values(result)
    assert _check_items(result, outs, output_names)
    assert all(output_names[i] in result.names()[i] for i in range(0, len(output_names)))

    return True


@pytest.mark.parametrize("is_direct", [True, False])
def test_ovdict_assign(device, is_direct):
    result, _ = _get_ovdict(device, multi_output=False, direct_infer=is_direct)

    with pytest.raises(TypeError) as e:
        result["some_name"] = 99
    assert "'OVDict' object does not support item assignment" in str(e.value)


@pytest.mark.parametrize("is_direct", [True, False])
def test_ovdict_single_output_basic(device, is_direct):
    result, obj = _get_ovdict(device, multi_output=False, direct_infer=is_direct)

    assert isinstance(result, OVDict)
    if isinstance(obj, (InferRequest, CompiledModel)):
        assert _check_dict(result, obj)
    else:
        raise TypeError("Unknown `obj` type!")


@pytest.mark.parametrize("is_direct", [True, False])
def test_ovdict_wrong_key_type(device, is_direct):
    result, _ = _get_ovdict(device, multi_output=False, direct_infer=is_direct)

    with pytest.raises(TypeError) as e:
        _ = result[2.0]
    assert "Unknown key type: <class 'float'>" in str(e.value)


@pytest.mark.parametrize("is_direct", [True, False])
def test_ovdict_single_output_noname(device, is_direct):
    result, obj = _get_ovdict(
        device,
        multi_output=False,
        direct_infer=is_direct,
        output_names=[],
    )

    assert isinstance(result, OVDict)

    outs = obj.model_outputs if isinstance(obj, InferRequest) else obj.outputs

    assert isinstance(result[outs[0]], np.ndarray)
    assert isinstance(result[0], np.ndarray)

    with pytest.raises(KeyError) as e0:
        _ = result["some_name"]
    assert "some_name" in str(e0.value)

    # Check if returned names are tuple with one default name set
    assert len(result.names()) == 1
    assert result.names()[0] != set()


@pytest.mark.parametrize("is_direct", [True, False])
def test_ovdict_single_output_wrongname(device, is_direct):
    result, obj = _get_ovdict(
        device,
        multi_output=False,
        direct_infer=is_direct,
        output_names=["output_21"],
    )

    assert isinstance(result, OVDict)

    outs = obj.model_outputs if isinstance(obj, InferRequest) else obj.outputs

    assert isinstance(result[outs[0]], np.ndarray)
    assert isinstance(result[0], np.ndarray)

    with pytest.raises(KeyError) as e:
        _ = result["output_37"]
    assert "output_37" in str(e.value)

    with pytest.raises(KeyError) as e:
        _ = result[6]
    assert "6" in str(e.value)


@pytest.mark.parametrize("is_direct", [True, False])
@pytest.mark.parametrize("use_function", [True, False])
def test_ovdict_single_output_dict(device, is_direct, use_function):
    result, obj = _get_ovdict(
        device,
        multi_output=False,
        direct_infer=is_direct,
    )

    assert isinstance(result, OVDict)

    outs = obj.model_outputs if isinstance(obj, InferRequest) else obj.outputs
    native_dict = result.to_dict() if use_function else dict(result)

    assert issubclass(type(native_dict), dict)
    assert not isinstance(native_dict, OVDict)
    assert isinstance(native_dict[outs[0]], np.ndarray)

    with pytest.raises(KeyError) as e:
        _ = native_dict["output_0"]
    assert "output_0" in str(e.value)

    with pytest.raises(KeyError) as e:
        _ = native_dict[0]
    assert "0" in str(e.value)


@pytest.mark.parametrize("is_direct", [True, False])
def test_ovdict_multi_output_basic(device, is_direct):
    output_names = ["output_0", "output_1", "output_2", "output_3", "output_4"]
    result, obj = _get_ovdict(
        device,
        multi_output=True,
        direct_infer=is_direct,
        output_names=output_names,
    )

    assert isinstance(result, OVDict)
    if isinstance(obj, (InferRequest, CompiledModel)):
        assert _check_dict(result, obj, output_names)
    else:
        raise TypeError("Unknown `obj` type!")


@pytest.mark.parametrize("is_direct", [True, False])
@pytest.mark.parametrize("use_function", [True, False])
def test_ovdict_multi_output_tuple0(device, is_direct, use_function):
    output_names = ["output_0", "output_1"]
    result, obj = _get_ovdict(
        device,
        input_shape=(1, 10),
        multi_output=True,
        direct_infer=is_direct,
        split_num=2,
        output_names=output_names,
    )

    out0, out1 = None, None
    if use_function:
        assert isinstance(result.to_tuple(), tuple)
        out0, out1 = result.to_tuple()
    else:
        out0, out1 = result.values()

    assert out0 is not None
    assert out1 is not None
    assert isinstance(out0, np.ndarray)
    assert isinstance(out1, np.ndarray)

    outs = obj.model_outputs if isinstance(obj, InferRequest) else obj.outputs

    assert np.equal(result[outs[0]], out0).all()
    assert np.equal(result[outs[1]], out1).all()
