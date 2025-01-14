# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestInstanceNormalization(OnnxRuntimeLayerTest):
    def create_net(self, shape, epsilon, precision, ir_version):
        """
            ONNX net                                     IR net

            Input->InstanceNormalization->Output   =>    Input->MVN->ScaleShift(Power)
        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

        scale_const = np.random.randn(shape[1]).astype(float)
        bias_const = np.random.randn(shape[1]).astype(float)

        node_scale_def = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['scale'],
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

        args = dict()
        if epsilon:
            args['epsilon'] = epsilon
        node_def = helper.make_node(
            'InstanceNormalization',
            inputs=['input', 'scale', 'bias'],
            outputs=['output'],
            **args
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_scale_def, node_bias_def, node_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #
        ref_net = None

        return onnx_net, ref_net

    test_data_precommit = [
        dict(shape=[1, 1, 4, 6], epsilon=0.001),
        dict(shape=[1, 1, 2, 4, 6], epsilon=0.001)]

    test_data = [
        dict(shape=[1, 1, 4, 6], epsilon=None),
        dict(shape=[1, 1, 4, 6], epsilon=0.001),
        dict(shape=[1, 2, 4, 6], epsilon=None),
        dict(shape=[1, 2, 4, 6], epsilon=0.001),
        dict(shape=[2, 3, 4, 6], epsilon=None),
        dict(shape=[2, 3, 4, 6], epsilon=0.001),
        dict(shape=[1, 1, 2, 4, 6], epsilon=None),
        dict(shape=[1, 1, 2, 4, 6], epsilon=0.001),
        dict(shape=[1, 2, 4, 6, 6], epsilon=None),
        dict(shape=[1, 2, 4, 6, 6], epsilon=0.001),
        dict(shape=[2, 3, 4, 6, 6], epsilon=None),
        dict(shape=[2, 3, 4, 6, 6], epsilon=0.001)]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_instance_normalization(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_instance_normalization(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
