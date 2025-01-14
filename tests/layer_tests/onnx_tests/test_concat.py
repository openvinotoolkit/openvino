# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestConcat(OnnxRuntimeLayerTest):
    # TODO Add test with default values (axis=0)
    def create_concat_net_const(self, input_shape, output_shape, axis, ir_version):
        """
            ONNX net                                                 IR net

            Input(const)----->Concat--------->Concat->Output   =>    Input--->Concat
            Input(const)-----'               '                        Const---'
                                      Input-'
        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto
        import numpy as np

        concat_axis = 0
        concat_output_shape = output_shape.copy()
        concat_output_shape[concat_axis] *= 2

        const_number = np.prod(input_shape)
        constant = np.random.randint(-127, 127, const_number).astype(float)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, output_shape)

        # Output for concat
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, concat_output_shape)

        node_const1_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const1'],
            value=helper.make_tensor(
                name='const1_tensor',
                data_type=TensorProto.FLOAT,
                dims=input_shape,
                vals=constant,
            ),
        )

        node_const2_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const2'],
            value=helper.make_tensor(
                name='const2_tensor',
                data_type=TensorProto.FLOAT,
                dims=input_shape,
                vals=constant,
            ),
        )

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['const1', 'const2'],
            outputs=['output_concat'],
            axis=axis
        )

        node_dyn_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['input', 'output_concat'],
            outputs=['output'],
            axis=concat_axis
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const1_def, node_const2_def, node_concat_def, node_dyn_concat_def],
            'test_concat_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_split_model')

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        constant_reshape = np.reshape(constant, input_shape)
        constant_reshape = np.concatenate([constant_reshape, constant_reshape], axis=axis)

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': output_shape, 'kind': 'data'},
                'input_const_data': {'kind': 'data', 'value': constant_reshape.flatten()},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': output_shape, 'value': None, 'kind': 'data'},
                'concat': {'kind': 'op', 'type': 'Concat', 'axis': concat_axis},
                'concat_data': {'shape': concat_output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_const_data', 'const'),
                                   ('const', 'const_data'),
                                   ('input_data', 'concat'),
                                   ('const_data', 'concat'),
                                   ('concat', 'concat_data'),
                                   ('concat_data', 'result')
                                   ])

        return onnx_net, ref_net

    def create_concat_net(self, input_shape, output_shape, axis, input_names, ir_version):
        """
            ONNX net                                                 IR net

            Input1----->Concat------>Output   =>    Input1--->Concat------>Output
            Input2-----'                            Input2---'
            Input3-----'                            Input3---'
            ...                                     ...
        """
        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        shape = input_shape
        inputs_list = []
        for input_name in input_names:
            inputs_list.append(helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape))

        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        node = onnx.helper.make_node('Concat', inputs=input_names, outputs=['output'], axis=axis)

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node],
            'concat_model',
            inputs_list,
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_concat_model')

        ref_net = None

        return onnx_net, ref_net

    test_data_3D = [
        dict(input_shape=[1, 50, 50],
             output_shape=[2, 50, 50],
             axis=0),

        dict(input_shape=[2, 50, 50],
             output_shape=[2, 100, 50],
             axis=1),

        dict(input_shape=[4, 50, 50],
             output_shape=[4, 50, 100],
             axis=2),
    ]

    test_data_4D_precommit = [
        dict(input_shape=[1, 32, 800, 800],
             output_shape=[2, 32, 800, 800],
             axis=0)
    ]

    test_data_4D = [
        dict(input_shape=[1, 32, 800, 800],
             output_shape=[2, 32, 800, 800],
             axis=0),

        dict(input_shape=[4, 32, 80, 80],
             output_shape=[4, 64, 80, 80],
             axis=1),

        dict(input_shape=[2, 21, 80, 80],
             output_shape=[2, 21, 160, 80],
             axis=2),

        dict(input_shape=[3, 21, 80, 80],
             output_shape=[3, 21, 80, 160],
             axis=3),
    ]

    test_data_5D_precommit = [
        dict(input_shape=[1, 50, 50, 80, 60],
             output_shape=[2, 50, 50, 80, 60],
             axis=0),

        dict(input_shape=[1, 50, 50, 80, 60],
             output_shape=[1, 50, 50, 80, 120],
             axis=4),
    ]

    test_data_5D = [
        dict(input_shape=[1, 50, 50, 80, 60],
             output_shape=[2, 50, 50, 80, 60],
             axis=0),

        dict(input_shape=[1, 50, 50, 80, 60],
             output_shape=[1, 100, 50, 80, 60],
             axis=1),

        dict(input_shape=[1, 50, 50, 80, 60],
             output_shape=[1, 50, 100, 80, 60],
             axis=2),

        dict(input_shape=[1, 50, 50, 80, 60],
             output_shape=[1, 50, 50, 160, 60],
             axis=3),

        dict(input_shape=[1, 50, 50, 80, 60],
             output_shape=[1, 50, 50, 80, 120],
             axis=4),
    ]

    test_concat_inputs_order_params = [
        dict(input_shape=[6],
             output_shape=[30],
             axis=0,
             input_names=['a', 't', 'm', 'p', 'e']),
        dict(input_shape=[5, 2],
             output_shape=[5, 8],
             axis=1,
             input_names=['inp2', 'inp1', 'inp5', 'inp4']),
        dict(input_shape=[6, 2, 5, 3],
             output_shape=[6, 2, 20, 3],
             axis=2,
             input_names=['n', 's', 'c', 'x']),
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_concat_3D_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net_const(**params, ir_version=ir_version), ie_device,
                   precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D_precommit)
    @pytest.mark.precommit
    def test_concat_4D_const_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net_const(**params, ir_version=ir_version), ie_device,
                   precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_concat_4D_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net_const(**params, ir_version=ir_version), ie_device,
                   precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D_precommit)
    @pytest.mark.nightly
    def test_concat_5D_const_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net_const(**params, ir_version=ir_version), ie_device,
                   precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_concat_5D_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net_const(**params, ir_version=ir_version), ie_device,
                   precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_concat_inputs_order_params)
    @pytest.mark.nightly
    def test_concat_inputs_order(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net(**params, ir_version=ir_version), ie_device=ie_device,
                   precision=precision, ir_version=ir_version, temp_dir=temp_dir,
                   input_names=params['input_names'])
