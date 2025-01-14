# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestDequantizeLinear(OnnxRuntimeLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(np.iinfo(self.inp_type).min,
                                                   np.iinfo(self.inp_type).max,
                                                   inputs_dict[input],
                                                   dtype=self.inp_type)
        return inputs_dict

    def create_dequanize_linear(self, shape, y_scale: np.array, y_zero_point=None, axis=None,
                                opset=10, ir_version='10'):
        """
            ONNX net                              IR net

            Input->DequantizeLinear->Output   =>    Input->Sub->Mul

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        self.inp_type = y_zero_point.dtype if y_zero_point is not None else np.uint8
        onnx_type = TensorProto.UINT8 if self.inp_type == np.uint8 else TensorProto.INT8
        input = helper.make_tensor_value_info('input', onnx_type, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

        nodes = []
        inputs = ['input', 'y_scale']
        node_scale_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['y_scale'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=y_scale.shape,
                vals=y_scale.flatten(),
            ),
        )
        nodes.append(node_scale_def)

        if y_zero_point is not None:
            node_zero_point_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['y_zero_point'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=onnx_type,
                    dims=y_zero_point.shape,
                    vals=y_zero_point.flatten(),
                ),
            )
            inputs.append('y_zero_point')
            nodes.append(node_zero_point_def)
        args = dict()
        if axis is not None:
            args['axis'] = axis
        node_def = onnx.helper.make_node(
            'DequantizeLinear',
            inputs=inputs,
            outputs=['output'],
            **args
        )
        nodes.append(node_def)

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            nodes,
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model',
                                     opset_imports=[helper.make_opsetid("", opset)])
        onnx.checker.check_model(onnx_net)

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        nodes_attributes = {
            'input': {'kind': 'op', 'type': 'Parameter'},
            'input_data': {'shape': shape, 'kind': 'data'},
            'input_scale_data': {'kind': 'data', 'value': y_scale},
            'scale_const': {'kind': 'op', 'type': 'Const'},
            'scale_data': {'shape': np.ones(len(shape)), 'kind': 'data'},
            'mul': {'kind': 'op', 'type': 'Multiply'},
            'mul_data': {'shape': shape, 'kind': 'data'},
            'result': {'kind': 'op', 'type': 'Result'}
        }
        edges = [('input', 'input_data'),
                 ('input_data', 'mul'),
                 ('input_scale_data', 'scale_const'),
                 ('scale_const', 'scale_data'),
                 ('scale_data', 'mul'),
                 ('mul', 'mul_data')]
        if y_zero_point is not None:
            nodes_attributes.update({
                'input_zero_data': {'kind': 'data', 'value': -y_scale * y_zero_point},
                'zero_const': {'kind': 'op', 'type': 'Const'},
                'zero_data': {'shape': np.ones(len(shape)), 'kind': 'data'},
                'sub': {'kind': 'op', 'type': 'Add'},
                'sub_data': {'shape': shape, 'kind': 'data'},
            })
            edges.extend([('mul_data', 'sub'),
                          ('input_zero_data', 'zero_const'),
                          ('zero_const', 'zero_data'),
                          ('zero_data', 'sub'),
                          ('sub', 'sub_data'),
                          ('sub_data', 'result')])
        else:
            edges.append(('mul_data', 'result'))

        ref_net = None
        if check_ir_version(10, None, ir_version):
            ref_net = build_graph(nodes_attributes, edges)

        return onnx_net, ref_net

    test_data = [
        dict(shape=[8], y_scale=np.array(2, dtype=float),
             y_zero_point=np.array(128, dtype=np.uint8)),
        dict(shape=[8], y_scale=np.array(2, dtype=float),
             y_zero_point=np.array(1, dtype=np.int8)),
        dict(shape=[2, 4], y_scale=np.array(2, dtype=float),
             y_zero_point=np.array(128, dtype=np.uint8)),
        dict(shape=[2, 4], y_scale=np.array(2, dtype=float),
             y_zero_point=np.array(1, dtype=np.int8)),
        dict(shape=[2, 4, 6], y_scale=np.array(2, dtype=float),
             y_zero_point=np.array(128, dtype=np.uint8)),
        dict(shape=[2, 4, 6], y_scale=np.array(2, dtype=float),
             y_zero_point=np.array(1, dtype=np.int8)),
        dict(shape=[2, 4, 6, 8], y_scale=np.array(2, dtype=float),
             y_zero_point=np.array(128, dtype=np.uint8)),
        dict(shape=[2, 4, 6, 8], y_scale=np.array(2, dtype=float),
             y_zero_point=np.array(1, dtype=np.int8)),
        dict(shape=[2, 4, 6, 8, 10], y_scale=np.array(2, dtype=float),
             y_zero_point=np.array(128, dtype=np.uint8)),
        dict(shape=[2, 4, 6, 8, 10], y_scale=np.array(2, dtype=float),
             y_zero_point=np.array(1, dtype=np.int8)),
    ]
    test_data_def_zerop = [
        dict(shape=[8], y_scale=np.array(2, dtype=float)),
        dict(shape=[2, 4], y_scale=np.array(2, dtype=float)),
        dict(shape=[2, 4, 6], y_scale=np.array(2, dtype=float)),
        dict(shape=[2, 4, 6, 8], y_scale=np.array(2, dtype=float)),
        dict(shape=[2, 4, 6, 8, 10], y_scale=np.array(2, dtype=float)),
    ]

    test_data_axis = [
        dict(shape=[2, 4], y_scale=np.array([2, 2.5, 3, 2.3], dtype=float), axis=1),
        dict(shape=[2, 4], y_scale=np.array([2, 2.5, 3, 2.3], dtype=float),
             y_zero_point=np.array([128, 128, 128, 128], dtype=np.uint8), axis=1),
        dict(shape=[2, 4], y_scale=np.array([2, 2.5, 3, 2.3], dtype=float),
             y_zero_point=np.array([1, 1, 1, 1], dtype=np.int8), axis=1),
        dict(shape=[2, 4, 6], y_scale=np.array([2, 2.5, 3, 2.3], dtype=float), axis=1),
        dict(shape=[2, 4, 6], y_scale=np.array([2, 2.5, 3, 2.3], dtype=float),
             y_zero_point=np.array([128, 128, 128, 128], dtype=np.uint8), axis=1),
        dict(shape=[2, 4, 6], y_scale=np.array([2, 2.5, 3, 2.3], dtype=float),
             y_zero_point=np.array([1, 1, 1, 1], dtype=np.int8), axis=1),
        dict(shape=[2, 4, 6, 8], y_scale=np.array([2, 2.5, 3, 2.3], dtype=float), axis=1),
        dict(shape=[2, 4, 6, 8], y_scale=np.array([2, 2.5, 3, 2.3], dtype=float),
             y_zero_point=np.array([128, 128, 128, 128], dtype=np.uint8), axis=1),
        dict(shape=[2, 4, 6, 8], y_scale=np.array([2, 2.5, 3, 2.3], dtype=float),
             y_zero_point=np.array([1, 1, 1, 1], dtype=np.int8), axis=1),
        dict(shape=[2, 4, 6, 8, 10], y_scale=np.array([2, 2.5, 3, 2.3], dtype=float), axis=1),
        dict(shape=[2, 4, 6, 8, 10], y_scale=np.array([2, 2.5, 3, 2.3], dtype=float),
             y_zero_point=np.array([128, 128, 128, 128], dtype=np.uint8), axis=1),
        dict(shape=[2, 4, 6, 8, 10], y_scale=np.array([2, 2.5, 3, 2.3], dtype=float),
             y_zero_point=np.array([1, 1, 1, 1], dtype=np.int8), axis=1),
    ]

    @pytest.mark.parametrize("params", test_data_def_zerop)
    @pytest.mark.nightly
    def test_quantize_linear_def_zerop_opset10(self, params, ie_device, precision, ir_version,
                                               temp_dir):
        self._test(*self.create_dequanize_linear(**params, ir_version=ir_version), ie_device,
                   precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_quantize_linear_opset10(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_dequanize_linear(**params, ir_version=ir_version), ie_device,
                   precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data + test_data_def_zerop)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='DequantizeLinear-13 is unsupported in MO')
    def test_quantize_linear_opset13(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_dequanize_linear(**params, opset=13, ir_version=ir_version),
                   ie_device, precision,
                   ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_axis)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='DequantizeLinear-13 is unsupported in MO')
    def test_quantize_linear_axis_opset13(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_dequanize_linear(**params, opset=13, ir_version=ir_version),
                   ie_device, precision,
                   ir_version, temp_dir=temp_dir)
