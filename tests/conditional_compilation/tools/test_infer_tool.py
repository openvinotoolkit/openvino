# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Unit tests for infer_tool.py helpers.
"""
import sys
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import op

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_tool import input_preparation  # noqa: E402 pylint: disable=import-error,wrong-import-position


def _compile_model(shapes):
    """Build and compile a model with one Parameter/Result pair per shape in `shapes`."""
    params = []
    for i, shape in enumerate(shapes):
        param = op.Parameter(ov.Type.f32, ov.Shape(shape))
        param.get_output_tensor(0).set_names({f"input_{i}"})
        params.append(param)
    results = [ov.opset13.result(param) for param in params]
    model = ov.Model(results, params, "test_model")
    return ov.Core().compile_model(model, "CPU")


# Legacy IR with no "names" attribute on ports: a real, still-produceable case where
# a model input has no tensor names at all (unlike models built via the Python API,
# which always assign a default tensor name).
_IR_WITHOUT_PORT_NAMES = b"""<?xml version="1.0"?>
<net name="test_model" version="11">
    <layers>
        <layer id="0" name="input" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,3"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
"""


def test_input_preparation_single_input():
    """input_preparation() must work against a CompiledModel, not the legacy IE 1.0 API."""

    compiled_model = _compile_model([[1, 3, 4, 4]])

    feed_dict = input_preparation(compiled_model)

    assert set(feed_dict.keys()) == set(compiled_model.inputs)
    assert feed_dict[compiled_model.inputs[0]].shape == (1, 3, 4, 4)
    assert np.all(feed_dict[compiled_model.inputs[0]] == 1)


def test_input_preparation_multiple_inputs():
    compiled_model = _compile_model([[1, 3], [2, 5]])

    feed_dict = input_preparation(compiled_model)

    assert set(feed_dict.keys()) == set(compiled_model.inputs)
    assert feed_dict[compiled_model.inputs[0]].shape == (1, 3)
    assert feed_dict[compiled_model.inputs[1]].shape == (2, 5)


def test_input_preparation_input_without_tensor_names():
    """Ports aren't guaranteed to have tensor names (e.g. older/hand-crafted IRs)."""
    core = ov.Core()
    model = core.read_model(model=_IR_WITHOUT_PORT_NAMES, weights=b"")
    assert not model.inputs[0].get_names(), "test setup must produce a nameless port"
    compiled_model = core.compile_model(model, "CPU")

    feed_dict = input_preparation(compiled_model)

    assert set(feed_dict.keys()) == set(compiled_model.inputs)
    assert feed_dict[compiled_model.inputs[0]].shape == (1, 3)


def test_infer_with_port_keyed_feed_dict():
    """input_preparation()'s port-keyed dict must be directly usable for inference."""
    compiled_model = _compile_model([[1, 3]])

    result = compiled_model(input_preparation(compiled_model))

    assert result[compiled_model.output(0)].shape == (1, 3)


def test_infer_with_name_keyed_feed_dict():
    """A name-keyed dict is also valid for inference when tensor names are present -
    proving port-keying was chosen for robustness against nameless ports, not because
    name-keyed dicts are broken."""
    compiled_model = _compile_model([[1, 3]])

    feed_dict = {model_input.any_name: np.ones(shape=list(model_input.shape)) for model_input in compiled_model.inputs}
    result = compiled_model(feed_dict)

    assert result[compiled_model.output(0)].shape == (1, 3)


def test_infer_with_nameless_port_keyed_feed_dict():
    """Only port-keying (not name-keying) works for inputs without tensor names."""
    core = ov.Core()
    model = core.read_model(model=_IR_WITHOUT_PORT_NAMES, weights=b"")
    compiled_model = core.compile_model(model, "CPU")

    result = compiled_model(input_preparation(compiled_model))

    assert result[compiled_model.output(0)].shape == (1, 3)
