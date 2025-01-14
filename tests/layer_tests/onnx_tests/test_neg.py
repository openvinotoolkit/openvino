# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestNeg(OnnxRuntimeLayerTest):
    def create_neg(self, shape, ir_version):
        """
            ONNX net                   IR net

            Input->Neg->Output   =>    Input->Power(scale=-1, shift=0, power=1)

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

        node_reduce_mean_def = onnx.helper.make_node(
            'Neg',
            inputs=['input'],
            outputs=['output'],
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_reduce_mean_def],
            'test_neg_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_neg_model')

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'neg': {'kind': 'op', 'type': 'Negative'},
                'neg_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'neg'),
                                   ('neg', 'neg_data'),
                                   ('neg_data', 'result')
                                   ])

        return onnx_net, ref_net

    test_data_precommit = [dict(shape=[2, 3, 4]),
                           dict(shape=[1, 3, 124, 124])]

    test_data = [dict(shape=[1, 64]),
                 dict(shape=[2, 3, 4]),
                 dict(shape=[1, 3, 124, 124]),
                 ]

    @pytest.mark.parametrize('params', test_data_precommit)
    @pytest.mark.precommit
    def test_neg_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_neg(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize('params', test_data)
    @pytest.mark.nightly
    def test_neg(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_neg(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)
