# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestFusedGemm(OnnxRuntimeLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randn(*inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_net(self, shapeA, shapeB, shapeC, alpha, beta, trans_a, trans_b,
                   activation, activation_alpha, activation_beta, activation_gamma,
                   opset, ir_version, ):
        """
            ONNX net                    IR net

            Input->FusedGemm->Output   =>    Input->Gemm->activation
        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        max_len = max([len(shapeA), len(shapeB)])
        extended_shape1 = np.concatenate([np.ones(max_len - len(shapeA)), shapeA], axis=0)
        extended_shape2 = np.concatenate([np.ones(max_len - len(shapeB)), shapeB], axis=0)
        output_shape = np.concatenate(
            [np.maximum(*[extended_shape1[0:-2], extended_shape2[0:-2]]), [shapeA[-2], shapeB[-1]]],
            axis=0).astype(int).tolist()
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shapeA)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        _shapeB = shapeB.copy()
        if trans_b:
            _shapeB.reverse()
        const1 = np.random.ranf(_shapeB).astype(float)
        const2 = np.random.ranf(shapeC).astype(float)

        nodes = list()
        node_const1_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const1'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=const1.shape,
                vals=const1.flatten(),
            ),
        )
        nodes.append(node_const1_def)

        inputs = ['input', 'const1']

        if opset is None or opset < 11:
            node_const2_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['const2'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.FLOAT,
                    dims=const2.shape,
                    vals=const2.flatten(),
                ),
            )
            inputs.append('const2')
            nodes.append(node_const2_def)

        attrs = dict()
        if alpha:
            attrs['alpha'] = alpha
        if beta:
            attrs['beta'] = beta
        if trans_a:
            attrs['transA'] = trans_a
        if trans_b:
            attrs['transB'] = trans_b
        if activation:
            attrs['activation'] = activation
        if activation_alpha:
            attrs['activation_alpha'] = activation_alpha
        if activation_beta:
            attrs['activation_beta'] = activation_beta
        if activation_gamma:
            attrs['activation_gamma'] = activation_gamma
        node_def = onnx.helper.make_node(
            'FusedGemm',
            inputs=inputs,
            outputs=['output'],
            domain='com.microsoft',
            **attrs
        )
        nodes.append(node_def)

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            nodes,
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        args = dict(producer_name='test_model')
        if opset:
            args['opset_imports'] = [helper.make_opsetid("", opset)]
        onnx_net = onnx_make_model(graph_def, **args)

        ref_net = None

        return onnx_net, ref_net


    test_data = [
        dict(shapeA=[3, 6], shapeB=[6, 4], shapeC=[3, 4], activation='LeakyRelu', activation_alpha=0.01, activation_beta=None, activation_gamma=None),
  #      dict(shapeA=[3, 6], shapeB=[6, 4], shapeC=[3, 4], activation='Relu', activation_alpha=None, activation_beta=None, activation_gamma=None),
    ]

    test_data_bc = [
        dict(shapeA=[3, 6], shapeB=[6, 4], shapeC=[4])
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("alpha", [0.1, 2.0])
    @pytest.mark.parametrize("beta", [0.1, 2.0])
    @pytest.mark.parametrize("trans_a", [None])
    @pytest.mark.parametrize("trans_b", [None, 1])
    @pytest.mark.parametrize("opset", [None])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_fusedgemm(self, params, alpha, beta, trans_a, trans_b,
                       ie_device, precision, opset, ir_version, temp_dir):
        self._test(
            *self.create_net(params['shapeA'], params['shapeB'], params['shapeC'], alpha, beta,
                             trans_a, trans_b,
                             params['activation'], params['activation_alpha'],
                             params['activation_beta'], params['activation_gamma'],
                             opset, ir_version), ie_device, precision,
            ir_version,
            temp_dir=temp_dir, custom_eps=1e-2)
