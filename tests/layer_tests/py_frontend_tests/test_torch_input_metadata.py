# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

import numpy as np
import openvino as ov
import pytest
import torch

from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from openvino.tools.ovc.cli_parser import _InputCutInfo
from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import extract_input_info_from_example


class TimestepModel(torch.nn.Module):
    def forward(self, sample: torch.Tensor, timestep: torch.Tensor):
        return sample + timestep.unsqueeze(0).to(torch.int32)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('reverse_inputs', [False, True])
@pytest.mark.parametrize('explicit_shape', [False, True])
def test_torchscript_dict_example_matches_graph_inputs(dtype, saved, reverse_inputs, explicit_shape, tmp_path):
    model = torch.jit.script(TimestepModel().eval())
    if saved:
        path = tmp_path / 'model.pt'
        model.save(str(path))
        model = torch.jit.load(str(path))
    sample = torch.ones(2, dtype=torch.float32)
    timestep = torch.tensor(7.0, dtype=dtype)
    examples = {'timestep': timestep, 'sample': sample}
    names = [value.debugName() for value in list(model.inlined_graph.inputs())[1:]]
    shapes = [[2], []]
    inputs = list(zip(names, shapes)) if explicit_shape else names
    tensors = [sample.numpy(), timestep.numpy()]
    if reverse_inputs:
        inputs = inputs[::-1]
        tensors = tensors[::-1]

    converted = ov.convert_model(model, example_input=examples, input=inputs)

    expected_types = [ov.Type.f32, ov.Type.f16 if dtype == torch.float16 else ov.Type.f32]
    if reverse_inputs:
        expected_types = expected_types[::-1]
    assert [port.get_element_type() for port in converted.inputs] == expected_types
    assert examples['timestep'] is timestep
    assert examples['sample'] is sample
    compiled = ov.Core().compile_model(converted, 'CPU', {'INFERENCE_PRECISION_HINT': 'f32'})
    actual = next(iter(compiled(tensors).values()))
    np.testing.assert_array_equal(actual, model(sample, timestep).numpy())


@pytest.mark.parametrize('case', ['explicit_metadata', 'exact_priority', 'unrelated_name'])
def test_torchscript_example_metadata_precedence(case):
    model = torch.jit.script(TimestepModel().eval())
    sample = torch.ones(2, dtype=torch.float32)
    timestep = torch.tensor(7.0, dtype=torch.float16)
    examples = {'timestep': timestep, 'sample': sample}
    names = [value.debugName() for value in list(model.inlined_graph.inputs())[1:]]
    descs = [_InputCutInfo(name, None, None, None) for name in names]
    if case == 'explicit_metadata':
        descs[0] = _InputCutInfo(names[0], ov.PartialShape([8]), ov.Type.f64, 5)
    elif case == 'exact_priority':
        examples[names[0]] = examples.pop('sample').to(torch.int64)
    else:
        descs[0] = _InputCutInfo('unrelated.1', None, None, None)
    decoder = TorchScriptPythonDecoder(model, example_input=examples)
    args = Namespace(input_model=decoder, example_input=examples, input=descs)

    extract_input_info_from_example(args, args.input)

    assert decoder._example_input is examples
    assert args.input[1].type == ov.Type.f16
    assert args.input[1].shape.rank.get_length() == 0
    if case == 'explicit_metadata':
        assert args.input[0] == descs[0]
    elif case == 'exact_priority':
        assert args.input[0].type == ov.Type.i64
        assert args.input[0].shape.rank.get_length() == 1
    else:
        assert args.input[0].type is None
        assert args.input[0].shape is None


def timestep_function(timestep: torch.Tensor):
    return timestep.unsqueeze(0).to(torch.int32)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_torchscript_function_example_matches_graph_input(dtype):
    model = torch.jit.script(timestep_function)
    timestep = torch.tensor(7.0, dtype=dtype)
    name = next(model.inlined_graph.inputs()).debugName()

    converted = ov.convert_model(model, example_input={'timestep': timestep}, input=[(name, [])])

    assert converted.input().get_element_type() == (ov.Type.f16 if dtype == torch.float16 else ov.Type.f32)
    compiled = ov.Core().compile_model(converted, 'CPU')
    np.testing.assert_array_equal(next(iter(compiled([timestep.numpy()]).values())), np.array([7], dtype=np.int32))


def test_torchscript_alias_does_not_reuse_exact_match():
    model = torch.jit.script(TimestepModel().eval())
    examples = {'sample': torch.ones(2), 'timestep': torch.tensor(7.0)}
    decoder = TorchScriptPythonDecoder(model, example_input=examples)
    debug_name = list(model.inlined_graph.inputs())[1].debugName()
    descs = [_InputCutInfo(name, None, None, None) for name in ['sample', debug_name]]
    args = Namespace(input_model=decoder, example_input=examples, input=descs)

    extract_input_info_from_example(args, args.input)

    assert args.input[0].type == ov.Type.f32
    assert args.input[1].type is None
    assert args.input[1].shape is None
