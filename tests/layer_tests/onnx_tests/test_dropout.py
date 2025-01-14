# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestDropout(OnnxRuntimeLayerTest):
    def create_net(self, shape, ratio, ir_version, opset=None):
        """
            ONNX net                                IR net

            Input->Dropout->Sigmoid->Output   =>    Input->sigmoid

        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

        args = dict()
        if ratio:
            args['ratio'] = ratio
        if opset == 6:
            args['is_test'] = 1
        node_def = helper.make_node(
            'Dropout',
            inputs=['input'],
            outputs=['dropout'],
            **args
        )

        sigmoid_def = helper.make_node(
            'Sigmoid',
            inputs=['dropout'],
            outputs=['output']
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def, sigmoid_def],
            'test_model',
            [input],
            [output],
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

    def create_net_const(self, shape, ratio, ir_version, opset=None):
        """
            ONNX net                                           IR net

            Input->Concat(+dropout with const)->Output   =>    Input->Concat(+const)

        """

        from onnx import helper
        from onnx import TensorProto

        constant = np.random.randint(-127, 127, shape).astype(float)

        concat_axis = 0
        output_shape = shape.copy()
        output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        node_const_def = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const1'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=constant.shape,
                vals=constant.flatten(),
            ),
        )

        args = dict()
        if ratio:
            args['ratio'] = ratio
        if opset == 6:
            args['is_test'] = 1
        node_def = helper.make_node(
            'Dropout',
            inputs=['const1'],
            outputs=['dropout'],
            **args
        )

        node_concat_def = helper.make_node(
            'Concat',
            inputs=['input', 'dropout'],
            outputs=['output'],
            axis=concat_axis
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_def, node_concat_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        args = dict(producer_name='test_model')
        if opset:
            args['opset_imports'] = [helper.make_opsetid("", opset)]
        onnx_net = onnx_make_model(graph_def, **args)

        ref_net = None

        return onnx_net, ref_net

    test_data = [dict(shape=[10, 12], ratio=None),
                 dict(shape=[10, 12], ratio=0.7),
                 dict(shape=[8, 10, 12], ratio=None),
                 dict(shape=[8, 10, 12], ratio=0.7),
                 dict(shape=[6, 8, 10, 12], ratio=None),
                 dict(shape=[6, 8, 10, 12], ratio=0.7),
                 dict(shape=[4, 6, 8, 10, 12], ratio=None),
                 dict(shape=[4, 6, 8, 10, 12], ratio=0.7)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_dropout_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, opset=6, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_dropout(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_dropout_const_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, opset=6, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_dropout_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)
