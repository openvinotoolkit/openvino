# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestReshape(OnnxRuntimeLayerTest):
    def create_reshape_net(self, input_shape, output_shape, ir_version):
        """
            ONNX net                                  IR net

            Input->Reshape->Output   =>    Input->Reshape

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        node_shape_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['shape'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.INT64,
                dims=[len(output_shape)],
                vals=output_shape,
            ),
        )

        node_reshape_def = onnx.helper.make_node(
            'Reshape',
            inputs=['input', 'shape'],
            outputs=['output']
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_shape_def, node_reshape_def],
            'test_reshape_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_reshape_model')

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': input_shape, 'kind': 'data'},
                'input_data_1': {'shape': [len(output_shape)], 'value': output_shape,
                                 'kind': 'data'},
                'const_1': {'kind': 'op', 'type': 'Const'},
                'const_data_1': {'shape': [len(output_shape)], 'value': None, 'kind': 'data'},
                # 'value': output_shape,
                'reshape': {'kind': 'op', 'type': 'Reshape'},
                'reshape_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data_1', 'const_1'),
                                   ('const_1', 'const_data_1'),
                                   ('const_data_1', 'reshape'),
                                   ('input_data', 'reshape'),
                                   ('reshape', 'reshape_data'),
                                   ('reshape_data', 'result')
                                   ])

        return onnx_net, ref_net

    def create_reshape_net_const(self, input_shape, output_shape, ir_version):
        """
            ONNX net                                         IR net

            Input->Concat(+reshaped const)->Output   =>    Input->Concat(+const)

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

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, output_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, concat_output_shape)

        const_number = np.prod(input_shape)
        constant = np.random.randint(-127, 127, const_number).astype(float)

        node_const_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const1'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=input_shape,
                vals=constant,
            ),
        )

        node_shape_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['shape'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.INT64,
                dims=[len(output_shape)],
                vals=output_shape,
            ),
        )

        node_reshape_def = onnx.helper.make_node(
            'Reshape',
            inputs=['const1', 'shape'],
            outputs=['reshape1']
        )

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['input', 'reshape1'],
            outputs=['output'],
            axis=concat_axis
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_shape_def, node_reshape_def, node_concat_def],
            'test_reshape_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_reshape_model')

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': output_shape, 'kind': 'data'},
                'input_const_data': {'kind': 'data', 'value': constant},
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
                                   ('concat_data', 'result'),
                                   ])

        return onnx_net, ref_net

    test_data_5D = [dict(input_shape=[4, 6, 8, 10, 12], output_shape=[4, 6, 8, 120]),
                    dict(input_shape=[4, 6, 8, 10, 12], output_shape=[4, 6, 80, 12]),
                    dict(input_shape=[4, 6, 8, 10, 12], output_shape=[4, 48, 10, 12]),
                    dict(input_shape=[4, 6, 8, 10, 12], output_shape=[24, 8, 10, 12]),
                    dict(input_shape=[4, 6, 8, 10], output_shape=[2, 2, 6, 8, 10]),
                    dict(input_shape=[4, 6, 8, 10], output_shape=[4, 2, 3, 8, 10]),
                    dict(input_shape=[4, 6, 8, 10], output_shape=[4, 6, 2, 4, 10]),
                    dict(input_shape=[4, 6, 8, 10], output_shape=[4, 6, 8, 2, 5])]

    test_data_5D_precommit = [dict(input_shape=[2, 4, 6, 8, 10], output_shape=[8, 6, 8, 10])]

    test_data_4D = [dict(input_shape=[4, 6, 8, 10], output_shape=[24, 8, 10]),
                    dict(input_shape=[4, 6, 8, 10], output_shape=[4, 48, 10]),
                    dict(input_shape=[4, 6, 8, 10], output_shape=[4, 6, 80]),
                    dict(input_shape=[4, 6, 8, 10], output_shape=[192, 10]),
                    dict(input_shape=[4, 6, 8, 10], output_shape=[4, 480]),
                    dict(input_shape=[4, 6, 8], output_shape=[2, 2, 6, 8]),
                    dict(input_shape=[4, 6, 8], output_shape=[4, 2, 3, 8]),
                    dict(input_shape=[4, 6, 8], output_shape=[4, 6, 2, 4]),
                    dict(input_shape=[4, 6], output_shape=[2, 2, 2, 3])]

    test_data_4D_precommit = [dict(input_shape=[2, 4, 6, 8], output_shape=[48, 8])]

    test_data_3D = [dict(input_shape=[4, 6, 8], output_shape=[24, 8]),
                    dict(input_shape=[4, 6, 8], output_shape=[4, 48]),
                    dict(input_shape=[4, 6], output_shape=[2, 2, 6]),
                    dict(input_shape=[4, 6], output_shape=[4, 2, 3]),
                    dict(input_shape=[4, 6], output_shape=[2, 4, 3])]

    test_data_3D_precommit = [dict(input_shape=[2, 4, 6], output_shape=[8, 6])]

    @pytest.mark.parametrize("params", test_data_5D_precommit)
    @pytest.mark.precommit
    def test_reshape_5D_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reshape_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D_precommit)
    @pytest.mark.precommit
    def test_reshape_4D_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reshape_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_3D_precommit)
    @pytest.mark.precommit
    def test_reshape_3D_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reshape_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_reshape_5D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reshape_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_reshape_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reshape_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_reshape_3D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reshape_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_reshape_const_5D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reshape_net_const(**params, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_reshape_const_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reshape_net_const(**params, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_reshape_const_3D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reshape_net_const(**params, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)
