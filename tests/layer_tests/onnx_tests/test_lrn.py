# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestLRN(OnnxRuntimeLayerTest):
    def create_net(self, shape, alpha, beta, bias, size, ir_version):
        """
            ONNX net                   IR net

            Input->LRN->Output   =>    Input->Norm->Power

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

        args = dict(size=size)
        if alpha:
            args['alpha'] = alpha
        if beta:
            args['beta'] = beta
        if bias:
            args['bias'] = bias
        node_def = onnx.helper.make_node(
            'LRN',
            inputs=['input'],
            outputs=['output'],
            **args
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #   Create reference IR net
        if not alpha:
            alpha = 0.0001
        if not beta:
            beta = 0.75
        if not bias:
            bias = 1.0
        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'const_indata': {'value': [1], 'kind': 'data'},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': [1], 'kind': 'data'},
                'norm': {'kind': 'op', 'type': 'LRN', 'alpha': alpha / bias, 'beta': beta,
                         'bias': bias,
                         'size': size},  # 'region': 'across'
                'norm_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            edges = [('input', 'input_data'),
                     ('input_data', 'norm'),
                     ('const_indata', 'const'),
                     ('const', 'const_data'),
                     ('const_data', 'norm'),
                     ('norm', 'norm_data'),
                     ('norm_data', 'result')
                     ]

            ref_net = build_graph(nodes_attributes, edges)

        return onnx_net, ref_net

    test_data_precommit = [
        dict(shape=[2, 12], alpha=None, beta=None, bias=None, size=1),
        pytest.param(dict(shape=[2, 3, 12], alpha=0.0002, beta=0.5, bias=2.0, size=3),
                     marks=pytest.mark.skip(reason="Skipped until fixed")),
        dict(shape=[2, 3, 12], alpha=0.0002, beta=0.5, bias=2.0, size=3),
        dict(shape=[2, 3, 12], alpha=0.0002, beta=0.5, bias=2.0, size=3)]

    test_data = [
        dict(shape=[2, 12], alpha=None, beta=None, bias=None, size=1),
        dict(shape=[2, 12], alpha=0.0002, beta=0.5, bias=2.0, size=1),
        dict(shape=[2, 3, 12], alpha=None, beta=None, bias=None, size=3),
        dict(shape=[2, 3, 12], alpha=0.0002, beta=0.5, bias=2.0, size=1),
        dict(shape=[2, 3, 12], alpha=0.0002, beta=0.5, bias=2.0, size=3),
        dict(shape=[2, 3, 8, 10, 12], alpha=None, beta=None, bias=None, size=3),
        dict(shape=[2, 3, 8, 10, 12], alpha=0.0002, beta=0.5, bias=2.0, size=1),
        dict(shape=[2, 3, 8, 10, 12], alpha=0.0002, beta=0.5, bias=2.0, size=3)]

    test_data_4D = [
        dict(shape=[2, 3, 10, 12], alpha=None, beta=None, bias=None, size=3),
        dict(shape=[2, 3, 10, 12], alpha=0.0002, beta=0.5, bias=2.0, size=1),
        dict(shape=[2, 3, 10, 12], alpha=0.0002, beta=0.5, bias=2.0, size=3)]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_lrn_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        # onnxruntime only supports 4D tensors for LRN
        self.skip_framework = True
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_lrn(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip('GREEN_SUITE')
        # onnxruntime only supports 4D tensors for LRN
        self.skip_framework = True
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_lrn_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self.skip_framework = False
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)
