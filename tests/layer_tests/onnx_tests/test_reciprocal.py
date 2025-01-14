# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestReciprocal(OnnxRuntimeLayerTest):
    def create_net(self, shape, ir_version):
        """
            ONNX net                              IR net

            Input+258->Reciprocal->Output   =>    Input->Power
        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

        # adding 258 is needed to avoid division by zero
        node_const_def = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=[1],
                vals=[258],
            ),
        )

        node_add_def = helper.make_node(
            'Add',
            inputs=['input', 'const'],
            outputs=['add']
        )

        node_def = helper.make_node(
            'Reciprocal',
            inputs=['add'],
            outputs=['output']
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_add_def, node_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #   Create reference IR net

        ref_net = None

        return onnx_net, ref_net

    def create_net_const(self, shape, precision, ir_version):
        """
            ONNX net                                            IR net

            Input->Concat with reciprocal consts->Output   =>   Input->Concat
        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        concat_axis = 0
        output_shape = list(shape)
        output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        const = np.random.randint(1, 256, shape).astype(float)

        node_const_def = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=const.shape,
                vals=const.flatten(),
            ),
        )

        node_def = helper.make_node(
            'Reciprocal',
            inputs=['const'],
            outputs=['node_out']
        )

        node_concat_def = helper.make_node(
            'Concat',
            inputs=['input', 'node_out'],
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
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #   Create reference IR net

        constant_calculated = 1 / const
        if precision == 'FP16':
            constant_calculated = constant_calculated.astype(np.float16)

        ref_net = None

        return onnx_net, ref_net

    test_data_precommit = [
        dict(shape=[2, 4]),
        dict(shape=[2, 4, 6, 8])]

    test_data = [
        dict(shape=[4, 6]),
        dict(shape=[4, 6, 8]),
        dict(shape=[4, 6, 8, 10]),
        dict(shape=[4, 6, 8, 10, 12])]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_reciprocal_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_reciprocal(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_reciprocal_const_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_reciprocal_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
