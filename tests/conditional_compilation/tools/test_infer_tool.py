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
from infer_tool import input_preparation, result_to_named_dict  # noqa: E402 pylint: disable=import-error,wrong-import-position,wrong-import-order

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
    """input_preparation() must key every input, not just the first one."""

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


def test_result_to_named_dict_uses_real_names():
    """Outputs with tensor names should keep those names as dict keys."""
    compiled_model = _compile_model([[1, 3]])

    result = compiled_model(input_preparation(compiled_model))
    named_result = result_to_named_dict(result)

    assert set(named_result.keys()) == {compiled_model.output(0).any_name}


def test_result_to_named_dict_falls_back_for_nameless_output():
    """Outputs without tensor names get a deterministic positional fallback name."""
    core = ov.Core()
    model = core.read_model(model=_IR_WITHOUT_PORT_NAMES, weights=b"")
    assert not model.outputs[0].get_names(), "test setup must produce a nameless output"
    compiled_model = core.compile_model(model, "CPU")

    result = compiled_model(input_preparation(compiled_model))
    named_result = result_to_named_dict(result)

    assert set(named_result.keys()) == {"output_0"}


# Two-output IR: output 0 has no tensor names, output 1 is explicitly named
# "output_0" - the same string the fallback would generate for output 0.
_IR_WITH_COLLIDING_OUTPUT_NAMES = b"""<?xml version="1.0"?>
<net name="test_model" version="11">
    <layers>
        <layer id="0" name="input0" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="input1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,1"/>
            <output>
                <port id="0" precision="FP32" names="output_0">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="output0" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
        <layer id="3" name="output1" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
"""


def test_result_to_named_dict_avoids_collision_with_real_name():
    """A real output name matching the fallback pattern must not overwrite another output."""
    core = ov.Core()
    model = core.read_model(model=_IR_WITH_COLLIDING_OUTPUT_NAMES, weights=b"")
    assert not model.outputs[0].get_names() and model.outputs[1].get_names() == {
        "output_0"
    }, "test setup must produce a nameless output colliding with a named one"
    compiled_model = core.compile_model(model, "CPU")

    result = compiled_model(
        {
            compiled_model.inputs[0]: np.array([[1.0]], dtype=np.float32),
            compiled_model.inputs[1]: np.array([[2.0]], dtype=np.float32),
        }
    )
    named_result = result_to_named_dict(result)

    assert len(named_result) == 2
    assert sorted(v.item() for v in named_result.values()) == [1.0, 2.0]
