# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np
import pytest
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestTranspose(OnnxRuntimeLayerTest):
    def create_net(self, shape, perm, ir_version):
        """
            ONNX net                                  IR net

            Input->Transpose->Sigmoid->Output   =>    Input->Permute->sigmoid

        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        output_shape = np.transpose(np.ones(shape), perm).shape
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        args = dict()
        if perm:
            args['perm'] = perm
        node_def = helper.make_node(
            'Transpose',
            inputs=['input'],
            outputs=['transpose'],
            **args
        )

        sigmoid_def = helper.make_node(
            'Sigmoid',
            inputs=['transpose'],
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
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #

        ref_net = None
        if not perm:
            perm = list(reversed(range(len(shape))))

        return onnx_net, ref_net

    def create_net_const(self, shape, perm, ir_version):
        """
            ONNX net                                         IR net

            Input->Concat(+transposed const)->Output   =>    Input->Concat(+const)

        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        constant = np.random.randint(-127, 127, shape).astype(float)
        constant_transposed = np.transpose(constant, perm)

        concat_axis = 0
        input_shape = list(constant_transposed.shape)
        output_shape = input_shape.copy()
        output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
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
        if perm:
            args['perm'] = perm
        node_def = helper.make_node(
            'Transpose',
            inputs=['const1'],
            outputs=['transpose'],
            **args
        )

        node_concat_def = helper.make_node(
            'Concat',
            inputs=['input', 'transpose'],
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

        #
        #   Create reference IR net
        #

        ref_net = None

        return onnx_net, ref_net

    test_data_precommit = [dict(shape=[4, 6, 8, 10, 12], perm=None),
                           dict(shape=[8, 10, 12], perm=[2, 1, 0]),
                           dict(shape=[6, 8, 10, 12], perm=[0, 3, 1, 2]),
                           dict(shape=[4, 6, 8, 10, 12], perm=[1, 0, 4, 3, 2])]

    test_data = [dict(shape=[10, 12], perm=None),
                 dict(shape=[8, 10, 12], perm=None),
                 dict(shape=[6, 8, 10, 12], perm=None),
                 dict(shape=[4, 6, 8, 10, 12], perm=None)]

    for shape in [[10, 12], [8, 10, 12], [6, 8, 10, 12], [4, 6, 8, 10, 12]]:
        for perm in itertools.permutations(np.arange(len(shape))):
            test_data.append(dict(shape=shape, perm=list(perm)))

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_transpose_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_transpose(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.nightly
    def test_transpose_const_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_transpose_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)
