# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestBatchNormalization(OnnxRuntimeLayerTest):
    def create_net(self, shape, epsilon, precision, ir_version, opset=None):
        """
            ONNX net                                  IR net

            Input->BatchNormalization->Output   =>    Input->ScaleShift(Power)
        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

        scale_const = np.random.randint(128, 256, shape[1]).astype(np.float32) / 128.
        bias_const = np.random.randint(0, 128, shape[1]).astype(np.float32) / 128.
        mean_const = np.random.randint(-127, 127, shape[1]).astype(np.float32)
        var_const = np.random.randint(128, 256, shape[1]).astype(np.float32) / 128.

        node_scale_def = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['scale_const'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=scale_const.shape,
                vals=scale_const.flatten(),
            ),
        )

        node_bias_def = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['bias'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=bias_const.shape,
                vals=bias_const.flatten(),
            ),
        )

        node_mean_def = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['mean'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=mean_const.shape,
                vals=mean_const.flatten(),
            ),
        )

        node_var_def = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['var'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=var_const.shape,
                vals=var_const.flatten(),
            ),
        )

        args = dict(epsilon=epsilon)
        if opset == 6:
            args['is_test'] = 1;
        node_def = helper.make_node(
            'BatchNormalization',
            inputs=['input', 'scale_const', 'bias', 'mean', 'var'],
            outputs=['output'],
            **args
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_scale_def, node_bias_def, node_mean_def, node_var_def, node_def],
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

    test_data = [dict(shape=[1, 1, 4, 6], epsilon=0.001),
                 dict(shape=[1, 2, 4, 6], epsilon=0.001),
                 dict(shape=[2, 3, 4, 6], epsilon=0.001)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_bn(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_bn_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, precision=precision, opset=6, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_bn_opset7(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, precision=precision, opset=7, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
