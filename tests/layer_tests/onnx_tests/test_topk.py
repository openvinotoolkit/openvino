# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestTopK(OnnxRuntimeLayerTest):
    def create_net(self, shape, k, axis, ir_version, largest=None, sorted=None, opset=None):
        """
            ONNX net                    IR net

            Input->TopK->Output   =>    Input->TopK

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        output_shape = shape.copy()
        if axis is not None:
            output_shape[axis] = k
        else:
            output_shape[-1] = k
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        values = helper.make_tensor_value_info('cvalues', TensorProto.FLOAT, output_shape)
        indices = helper.make_tensor_value_info('cindices', TensorProto.INT64, output_shape)

        const1 = np.ones(output_shape).astype(np.int64)
        const2 = np.ones(output_shape).astype(float)

        nodes = list()
        inputs = ['input']
        if opset > 9:
            node_k_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['k'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.INT64,
                    dims=[1],
                    vals=[k],
                ),
            )
            nodes.append(node_k_def)
            inputs.append('k')

        args = dict()
        if opset < 10:
            args['k'] = k
        if axis is not None:
            args['axis'] = axis
        if sorted is not None:
            args['sorted'] = sorted
        if largest is not None:
            args['largest'] = largest

        node_def = onnx.helper.make_node(
            'TopK',
            inputs=inputs,
            outputs=['values', 'indices'],
            **args
        )

        node_const1_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const1'],
            value=helper.make_tensor(
                name='const_tensor2',
                data_type=TensorProto.INT64,
                dims=const1.shape,
                vals=const1.flatten(),
            ),
        )

        node_add1_def = onnx.helper.make_node(
            'Add',
            inputs=['indices', 'const1'],
            outputs=['cindices']
        )

        node_const2_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const2'],
            value=helper.make_tensor(
                name='const_tensor3',
                data_type=TensorProto.FLOAT,
                dims=const2.shape,
                vals=const2.flatten(),
            ),
        )

        node_add2_def = onnx.helper.make_node(
            'Add',
            inputs=['values', 'const2'],
            outputs=['cvalues']
        )

        nodes.extend([node_def, node_const1_def, node_add1_def, node_const2_def, node_add2_def])

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            nodes,
            'test_model',
            [input],
            [values, indices],
        )

        # Create the model (ModelProto)
        args = dict(producer_name='test_model')
        if opset:
            args['opset_imports'] = [helper.make_opsetid("", opset)]
        onnx_net = onnx_make_model(graph_def, **args)

        #
        #   Create reference IR net
        #

        ref_net = None

        return onnx_net, ref_net

    test_data = [dict(shape=[10, 12], k=3, axis=0),
                 dict(shape=[10, 12], k=5, axis=1),
                 dict(shape=[8, 10, 12], k=3, axis=0),
                 dict(shape=[8, 10, 12], k=4, axis=1),
                 dict(shape=[8, 10, 12], k=5, axis=2),
                 dict(shape=[6, 8, 10, 12], k=3, axis=0),
                 dict(shape=[6, 8, 10, 12], k=4, axis=1),
                 dict(shape=[6, 8, 10, 12], k=5, axis=2),
                 dict(shape=[6, 8, 10, 12], k=6, axis=3),
                 dict(shape=[4, 6, 8, 10, 12], k=3, axis=0),
                 dict(shape=[4, 6, 8, 10, 12], k=4, axis=1),
                 dict(shape=[4, 6, 8, 10, 12], k=5, axis=2),
                 dict(shape=[4, 6, 8, 10, 12], k=6, axis=3),
                 dict(shape=[4, 6, 8, 10, 12], k=7, axis=4)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_topk_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, opset=6, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_topk_opset10(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'CPU':
            pytest.skip('GREEN_SUITE')
        self._test(*self.create_net(**params, opset=10, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("largest", [1, 0, None])
    @pytest.mark.parametrize("sorted", [1, 0, None])
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_topk_opset11(self, params, ie_device, precision, ir_version, largest, sorted, temp_dir):
        self._test(*self.create_net(**params, largest=largest, sorted=sorted,
                                    opset=11, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)
