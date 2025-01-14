# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestMatMul(OnnxRuntimeLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randn(*inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_net(self, shape1, shape2, precision, ir_version):
        """
            ONNX net                                 IR net

            Input->MatMul with const->Output   =>    Input->FullyConnected
        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        max_len = max([len(shape1), len(shape2)])
        extended_shape1 = np.concatenate([np.ones(max_len - len(shape1)), shape1], axis=0)
        extended_shape2 = np.concatenate([np.ones(max_len - len(shape2)), shape2], axis=0)
        output_shape = np.concatenate(
            [np.maximum(*[extended_shape1[0:-2], extended_shape2[0:-2]]), [shape1[-2], shape2[-1]]],
            axis=0).astype(int).tolist()
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape1)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        const = np.random.randn(*shape2).astype(np.float32)

        node_const_def = onnx.helper.make_node(
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

        node_def = onnx.helper.make_node(
            'MatMul',
            inputs=['input', 'const'],
            outputs=['mm_output']
        )

        # to avoid mapping problems
        node_elu_def = onnx.helper.make_node(
            'Elu',
            inputs=['mm_output'],
            outputs=['output']
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_def, node_elu_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #   Please, spesify 'type': 'Input' for inpit node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        if precision == 'FP16':
            const = const.astype(np.float16)
        ref_net = None

        return onnx_net, ref_net

    def create_dual_net(self, shape1, shape2, ir_version):
        """
            ONNX net                                 IR net

            Input->MatMul->Output   =>    Input->Concat
        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        max_len = max([len(shape1), len(shape2)])
        extended_shape1 = np.concatenate([np.ones(max_len - len(shape1)), shape1], axis=0)
        extended_shape2 = np.concatenate([np.ones(max_len - len(shape2)), shape2], axis=0)
        output_shape = np.concatenate(
            [np.maximum(*[extended_shape1[0:-2], extended_shape2[0:-2]]), [shape1[-2], shape2[-1]]],
            axis=0).astype(int).tolist()
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, shape1)
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, shape2)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        node_def = onnx.helper.make_node(
            'MatMul',
            inputs=['input1', 'input2'],
            outputs=['output']
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            'test_model',
            [input1, input2],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #   Please, spesify 'type': 'Input' for inpit node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return onnx_net, ref_net

    test_data = [
        dict(shape1=[4, 6], shape2=[6, 4]),
        dict(shape1=[1, 4, 6], shape2=[1, 6, 4]),
        dict(shape1=[2, 4, 6], shape2=[2, 6, 4]),
        dict(shape1=[1, 1, 4, 6], shape2=[1, 1, 6, 4]),
        dict(shape1=[1, 2, 4, 6], shape2=[1, 2, 6, 4]),
        dict(shape1=[2, 3, 4, 6], shape2=[2, 3, 6, 4]),
        dict(shape1=[2, 3, 4, 4, 6], shape2=[2, 3, 4, 6, 4])
    ]

    test_data_broadcasting = [
        dict(shape1=[1, 4, 6], shape2=[6, 4]),
        dict(shape1=[2, 4, 6], shape2=[6, 4]),
        dict(shape1=[2, 4, 6], shape2=[1, 6, 4]),
        dict(shape1=[1, 1, 4, 6], shape2=[6, 4]),
        dict(shape1=[1, 1, 4, 6], shape2=[1, 6, 4]),
        dict(shape1=[1, 2, 4, 6], shape2=[6, 4]),
        dict(shape1=[1, 2, 4, 6], shape2=[2, 6, 4]),
        dict(shape1=[2, 3, 4, 6], shape2=[6, 4]),
        dict(shape1=[2, 3, 4, 6], shape2=[3, 6, 4]),
        dict(shape1=[2, 3, 4, 6], shape2=[1, 3, 6, 4]),
        dict(shape1=[2, 3, 4, 4, 6], shape2=[6, 4]),
        dict(shape1=[2, 3, 4, 4, 6], shape2=[4, 6, 4]),
        dict(shape1=[2, 3, 4, 4, 6], shape2=[3, 4, 6, 4])
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_matmul(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_broadcasting)
    @pytest.mark.nightly
    def test_matmul_bc(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip('GREEN_SUITE')
        self._test(*self.create_net(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_dual_matmul(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_dual_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_broadcasting)
    @pytest.mark.nightly
    def test_dual_matmul_bc(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip('GREEN_SUITE')
        self._test(*self.create_dual_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)
