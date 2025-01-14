# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestScatters(OnnxRuntimeLayerTest):
    op = None

    def create_net(self, input_shape, indices_shape, updates_shape, output_shape,
                   axis, ir_version):
        """
            ONNX net                    IR net

            Input->Scatter->Output   =>    Parameter->ScatterElementsUpdate->Result

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        data = helper.make_tensor_value_info('data', TensorProto.FLOAT, input_shape)
        indices = helper.make_tensor_value_info('indices', TensorProto.INT64, indices_shape)
        updates = helper.make_tensor_value_info('updates', TensorProto.FLOAT, indices_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        params = {'axis': axis} if axis is not None else {}
        node_def = onnx.helper.make_node(
            self.op,
            inputs=['data', 'indices', 'updates'],
            outputs=['output'],
            **params,
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            'test_model',
            [data, indices, updates],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #

        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                # comparison in these tests starts from input node, as we have 3 of them IREngine gets confused
                # and takes the first input node in inputs list sorted by lexicographical order
                '1_input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': input_shape, 'kind': 'data'},

                '2_indices': {'kind': 'op', 'type': 'Parameter'},
                'indices_data': {'shape': indices_shape, 'kind': 'data'},

                '3_updates': {'kind': 'op', 'type': 'Parameter'},
                'updates_data': {'shape': updates_shape, 'kind': 'data'},

                'const_indata': {'kind': 'data',
                                 'value': np.int64(axis) if axis is not None else np.int64(0)},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'kind': 'data'},

                'node': {'kind': 'op', 'type': 'ScatterElementsUpdate'},
                'node_data': {'shape': output_shape, 'kind': 'data'},

                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [
                                      ('1_input', 'input_data'),
                                      ('input_data', 'node', {'in': 0}),
                                      ('2_indices', 'indices_data'),
                                      ('indices_data', 'node', {'in': 1}),
                                      ('3_updates', 'updates_data'),
                                      ('updates_data', 'node', {'in': 2}),
                                      ('const_indata', 'const'),
                                      ('const', 'const_data'),
                                      ('const_data', 'node', {'in': 3}),

                                      ('node', 'node_data'),
                                      ('node_data', 'result')
                                  ])
        return onnx_net, ref_net


test_data = [
    dict(input_shape=[1, 5], indices_shape=[1, 2], updates_shape=[1, 2],
         axis=1, output_shape=[1, 5]),
    dict(input_shape=[1, 256, 200, 272], indices_shape=[1, 256, 200, 272],
         updates_shape=[1, 256, 200, 272],
         axis=None, output_shape=[1, 256, 200, 272])]


class TestScatter(TestScatters):
    op = 'Scatter'

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_scatter(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)


class TestScatterElements(TestScatters):
    op = 'ScatterElements'

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_scatter_elements(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)
