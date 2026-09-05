# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model
from common.utils.common_utils import generate_ir_python_api


class TestIf(OnnxRuntimeLayerTest):
    @staticmethod
    def create_const(name, tensor_type, value):
        from onnx import helper
        from onnx import TensorProto

        if tensor_type == TensorProto.INT64:
            np_type = np.int64
        elif tensor_type == TensorProto.FLOAT:
            np_type = np.float32
        elif tensor_type == TensorProto.BOOL:
            np_type = bool
        else:
            return None

        value = np.array(value)
        return helper.make_node(
            'Constant',
            inputs=[],
            outputs=[name],
            value=helper.make_tensor(
                name=name + '_tensor',
                data_type=tensor_type,
                dims=value.shape,
                vals=value.flatten().astype(np_type),
            ),
        )

    def _prepare_input(self, inputs_dict, kwargs_to_prepare_input=None):
        rng = np.random.default_rng(42)
        params = kwargs_to_prepare_input or {}

        for input_name, shape in inputs_dict.items():
            if input_name in params:
                value = params[input_name]
                if isinstance(value, np.ndarray):
                    inputs_dict[input_name] = value
                else:
                    dtype = bool if input_name.endswith('cond') or input_name == 'cond' else np.float32
                    inputs_dict[input_name] = np.array(value, dtype=dtype)
            else:
                if input_name.endswith('cond') or input_name == 'cond':
                    inputs_dict[input_name] = np.array([True], dtype=bool)
                else:
                    inputs_dict[input_name] = rng.standard_normal(shape).astype(np.float32)
        return inputs_dict

    def _test_conversion_only(self, framework_model, precision, temp_dir):
        import openvino as ov

        model_path = self.produce_model_path(framework_model=framework_model, save_path=temp_dir)
        exit_code, stderr = generate_ir_python_api(input_model=model_path,
                                                   output_dir=temp_dir,
                                                   compress_to_fp16=(precision != 'FP32'))
        assert not exit_code, stderr
        ov.Core().read_model(Path(temp_dir, 'model.xml'))

    @staticmethod
    def _make_identity_branch(output_name):
        from onnx import helper
        from onnx import TensorProto

        branch_output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, None)
        branch_graph = helper.make_graph(
            [helper.make_node('Identity', inputs=['x'], outputs=[output_name])],
            output_name + '_graph',
            [],
            [branch_output],
        )
        return branch_graph

    @staticmethod
    def _make_unsqueeze_branch(output_name, axes_name='axes_unsqueeze'):
        from onnx import helper
        from onnx import TensorProto

        axes = TestIf.create_const(axes_name, TensorProto.INT64, np.array([0], dtype=np.int64))
        branch_output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, None)
        branch_graph = helper.make_graph(
            [axes, helper.make_node('Unsqueeze', inputs=['x', axes_name], outputs=[output_name])],
            output_name + '_graph',
            [],
            [branch_output],
        )
        return branch_graph

    def create_if_different_shapes(self):
        from onnx import helper
        from onnx import TensorProto

        x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
        cond = helper.make_tensor_value_info('cond', TensorProto.BOOL, [1])
        output = helper.make_tensor_value_info('output', TensorProto.INT64, None)

        then_axes = self.create_const('then_axes', TensorProto.INT64, np.array([0], dtype=np.int64))
        then_output = helper.make_tensor_value_info('then_output', TensorProto.INT64, None)
        then_branch = helper.make_graph(
            [
                then_axes,
                helper.make_node('Unsqueeze', inputs=['x', 'then_axes'], outputs=['then_tensor']),
                helper.make_node('Shape', inputs=['then_tensor'], outputs=['then_output']),
            ],
            'then_output_graph',
            [],
            [then_output],
        )

        else_axes = self.create_const('else_axes', TensorProto.INT64, np.array([0], dtype=np.int64))
        else_output = helper.make_tensor_value_info('else_output', TensorProto.INT64, None)
        else_branch = helper.make_graph(
            [
                else_axes,
                helper.make_node('Unsqueeze', inputs=['x', 'else_axes'], outputs=['else_tensor_0']),
                helper.make_node('Concat', inputs=['else_tensor_0', 'else_tensor_0'], outputs=['else_tensor'], axis=0),
                helper.make_node('Shape', inputs=['else_tensor'], outputs=['else_output']),
            ],
            'else_output_graph',
            [],
            [else_output],
        )

        if_node = helper.make_node('If', inputs=['cond'], outputs=['if_output'],
                                   then_branch=then_branch, else_branch=else_branch)
        res_node = helper.make_node('Identity', inputs=['if_output'], outputs=['output'])

        graph_def = helper.make_graph([if_node, res_node], 'if_different_shapes', [cond, x], [output])
        onnx_net = onnx_make_model(graph_def,
                                   producer_name='test_if_model',
                                   opset_imports=[helper.make_opsetid('', 16)])
        return onnx_net, None

    def create_if_unselected_invalid_transpose(self):
        from onnx import helper
        from onnx import TensorProto

        x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
        cond = helper.make_tensor_value_info('cond', TensorProto.BOOL, [1])
        rank_cond = helper.make_tensor_value_info('rank_cond', TensorProto.BOOL, [1])
        output = helper.make_tensor_value_info('output', TensorProto.INT64, None)

        then_axes = self.create_const('then_axes_unselected_output', TensorProto.INT64, np.array([0], dtype=np.int64))
        then_output = helper.make_tensor_value_info('then_output', TensorProto.INT64, None)
        then_branch = helper.make_graph(
            [
                then_axes,
                helper.make_node('Unsqueeze', inputs=['x', 'then_axes_unselected_output'], outputs=['then_tensor']),
                helper.make_node('Shape', inputs=['then_tensor'], outputs=['then_output']),
            ],
            'then_output_unselected_graph',
            [],
            [then_output],
        )
        inner_then = self._make_unsqueeze_branch('inner_then_output', axes_name='then_axes_unselected')
        inner_else = helper.make_graph(
            [helper.make_node('Identity', inputs=['x'], outputs=['inner_else_output'])],
            'inner_else_unselected_graph',
            [],
            [helper.make_tensor_value_info('inner_else_output', TensorProto.FLOAT, None)],
        )
        else_output = helper.make_tensor_value_info('else_output', TensorProto.INT64, None)
        else_branch = helper.make_graph(
            [
                helper.make_node('If', inputs=['rank_cond'], outputs=['rank_variant_output'],
                                 then_branch=inner_then, else_branch=inner_else),
                helper.make_node('Transpose', inputs=['rank_variant_output'], outputs=['transposed_output'], perm=[1, 0, 2]),
                helper.make_node('Shape', inputs=['transposed_output'], outputs=['else_output']),
            ],
            'if_else_unselected_dynamic_rank_transpose',
            [],
            [else_output],
        )

        if_node = helper.make_node('If', inputs=['cond'], outputs=['if_output'],
                                   then_branch=then_branch, else_branch=else_branch)
        res_node = helper.make_node('Identity', inputs=['if_output'], outputs=['output'])

        graph_def = helper.make_graph([if_node, res_node], 'if_unselected_dynamic_rank_transpose',
                                      [cond, rank_cond, x], [output])
        onnx_net = onnx_make_model(graph_def,
                                   producer_name='test_if_unselected_branch_model',
                                   opset_imports=[helper.make_opsetid('', 16)])
        return onnx_net, None

    def create_if_dynamic_rank_transpose(self):
        from onnx import helper
        from onnx import TensorProto

        x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
        cond = helper.make_tensor_value_info('cond', TensorProto.BOOL, [1])
        rank_cond = helper.make_tensor_value_info('rank_cond', TensorProto.BOOL, [1])
        output = helper.make_tensor_value_info('output', TensorProto.INT64, None)

        then_axes = self.create_const('then_axes_output', TensorProto.INT64, np.array([0], dtype=np.int64))
        then_output = helper.make_tensor_value_info('then_output', TensorProto.INT64, None)
        then_branch = helper.make_graph(
            [
                then_axes,
                helper.make_node('Unsqueeze', inputs=['x', 'then_axes_output'], outputs=['then_tensor']),
                helper.make_node('Shape', inputs=['then_tensor'], outputs=['then_output']),
            ],
            'then_output_graph',
            [],
            [then_output],
        )
        inner_then = self._make_unsqueeze_branch('inner_then_output', axes_name='then_axes')
        inner_else = helper.make_graph(
            [helper.make_node('Identity', inputs=['x'], outputs=['inner_else_output'])],
            'inner_else_graph',
            [],
            [helper.make_tensor_value_info('inner_else_output', TensorProto.FLOAT, None)],
        )
        else_output = helper.make_tensor_value_info('else_output', TensorProto.INT64, None)
        else_branch = helper.make_graph(
            [
                helper.make_node('If', inputs=['rank_cond'], outputs=['rank_variant_output'],
                                 then_branch=inner_then, else_branch=inner_else),
                helper.make_node('Transpose', inputs=['rank_variant_output'], outputs=['transposed_output'], perm=[1, 0, 2]),
                helper.make_node('Shape', inputs=['transposed_output'], outputs=['else_output']),
            ],
            'if_else_dynamic_rank_transpose',
            [],
            [else_output],
        )

        if_node = helper.make_node('If', inputs=['cond'], outputs=['if_output'],
                                   then_branch=then_branch, else_branch=else_branch)
        res_node = helper.make_node('Identity', inputs=['if_output'], outputs=['output'])

        graph_def = helper.make_graph([if_node, res_node], 'if_dynamic_rank_transpose', [cond, rank_cond, x], [output])
        onnx_net = onnx_make_model(graph_def,
                                   producer_name='test_if_dynamic_rank_model',
                                   opset_imports=[helper.make_opsetid('', 16)])
        return onnx_net, None

    @pytest.mark.precommit
    @pytest.mark.timeout(250)
    def test_if_different_shapes_precommit(self, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_if_different_shapes(),
                   ie_device,
                   precision,
                   ir_version,
                   temp_dir=temp_dir,
                   infer_timeout=150,
                   kwargs_to_prepare_input={'cond': np.array([False], dtype=bool)})

    @pytest.mark.precommit
    @pytest.mark.timeout(250)
    def test_if_unselected_invalid_transpose_precommit(self, ie_device, precision, ir_version, temp_dir):
        del ie_device, ir_version
        self._test_conversion_only(self.create_if_unselected_invalid_transpose()[0], precision, temp_dir)

    @pytest.mark.precommit
    @pytest.mark.timeout(250)
    def test_if_dynamic_rank_transpose_precommit(self, ie_device, precision, ir_version, temp_dir):
        del ie_device, ir_version
        self._test_conversion_only(self.create_if_dynamic_rank_transpose()[0], precision, temp_dir)
