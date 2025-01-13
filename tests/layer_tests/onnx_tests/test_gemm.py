# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import torch
from common.layer_test_class import CommonLayerTest
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestGemm(OnnxRuntimeLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randn(*inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_net(self, shapeA, shapeB, shapeC, alpha, beta, trans_a, trans_b, precision, opset,
                   ir_version, ):
        """
            ONNX net                    IR net

            Input->Gemm->Output   =>    Input->Concat
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
        node_def = onnx.helper.make_node(
            'Gemm',
            inputs=inputs,
            outputs=['output'],
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

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        if alpha:
            const1 *= alpha
        if beta:
            const2 *= beta
        if precision == 'FP16':
            const1 = const1.astype(np.float16)
            const2 = const2.astype(np.float16)
        if not trans_b:
            const1 = const1.transpose()

        ref_net = None

        return onnx_net, ref_net

    def create_net_double(self, shapeA, shapeB, shapeC, alpha, beta, trans_a, trans_b, precision,
                          ir_version):
        """
            ONNX net                    IR net

            Input->Gemm->Output   =>    Input->Concat
        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        _shapeA = shapeA.copy()
        if trans_a:
            _shapeA.reverse()
        _shapeB = shapeB.copy()
        if trans_b:
            _shapeB.reverse()

        max_len = max([len(shapeA), len(shapeB)])
        extended_shape1 = np.concatenate([np.ones(max_len - len(shapeA)), shapeA], axis=0)
        extended_shape2 = np.concatenate([np.ones(max_len - len(shapeB)), shapeB], axis=0)
        output_shape = np.concatenate(
            [np.maximum(*[extended_shape1[0:-2], extended_shape2[0:-2]]), [shapeA[-2], shapeB[-1]]],
            axis=0).astype(int).tolist()
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, _shapeA)
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, _shapeB)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        const = np.random.ranf(shapeC).astype(float)

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

        attrs = dict()
        if alpha:
            attrs['alpha'] = alpha
        if beta:
            attrs['beta'] = beta
        if trans_a:
            attrs['transA'] = trans_a
        if trans_b:
            attrs['transB'] = trans_b
        node_def = onnx.helper.make_node(
            'Gemm',
            inputs=['input1', 'input2', 'const'],
            outputs=['output'],
            **attrs
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_def],
            'test_model',
            [input1, input2],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        if precision == 'FP16':
            const = const.astype(np.float16)
        ref_net = None

        return onnx_net, ref_net

    test_data = [
        dict(shapeA=[3, 6], shapeB=[6, 4], shapeC=[3, 4])
    ]

    test_data_bc = [
        dict(shapeA=[3, 6], shapeB=[6, 4], shapeC=[4])
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("alpha", [None, 0.1, 2.0])
    @pytest.mark.parametrize("beta", [None, 0.1, 2.0])
    @pytest.mark.parametrize("trans_a", [None])
    @pytest.mark.parametrize("trans_b", [None, 1])
    @pytest.mark.parametrize("opset", [None, 11])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_gemm(self, params, alpha, beta, trans_a, trans_b, ie_device, precision, opset,
                  ir_version, temp_dir):
        self._test(
            *self.create_net(params['shapeA'], params['shapeB'], params['shapeC'], alpha, beta,
                             trans_a,
                             trans_b, precision, opset, ir_version), ie_device, precision,
            ir_version,
            temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_bc)
    @pytest.mark.parametrize("alpha", [None, 0.1, 2.0])
    @pytest.mark.parametrize("beta", [None, 0.1, 2.0])
    @pytest.mark.parametrize("trans_a", [None])  # transA is not supported
    @pytest.mark.parametrize("trans_b", [None, 1])
    @pytest.mark.parametrize("opset", [None, 11])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_gemm_bc(self, params, alpha, beta, trans_a, trans_b, ie_device, precision, opset,
                     ir_version, temp_dir):
        self._test(
            *self.create_net(params['shapeA'], params['shapeB'], params['shapeC'], alpha, beta,
                             trans_a,
                             trans_b, precision, opset, ir_version), ie_device, precision,
            ir_version,
            temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("alpha", [None, 0.1, 2.0])
    @pytest.mark.parametrize("beta", [None, 0.1, 2.0])
    @pytest.mark.parametrize("trans_a", [None, 1])
    @pytest.mark.parametrize("trans_b", [None, 1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_gemm_double(self, params, alpha, beta, trans_a, trans_b, ie_device, precision,
                         ir_version, temp_dir):
        self._test(
            *self.create_net_double(params['shapeA'], params['shapeB'], params['shapeC'], alpha,
                                    beta,
                                    trans_a, trans_b, precision, ir_version), ie_device, precision,
            ir_version,
            temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_bc)
    @pytest.mark.parametrize("alpha", [None, 0.1, 2.0])
    @pytest.mark.parametrize("beta", [None, 0.1, 2.0])
    @pytest.mark.parametrize("trans_a", [None, 1])
    @pytest.mark.parametrize("trans_b", [None, 1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_gemm_double_bc(self, params, alpha, beta, trans_a, trans_b, ie_device, precision,
                            ir_version, temp_dir):
        self._test(
            *self.create_net_double(params['shapeA'], params['shapeB'], params['shapeC'], alpha,
                                    beta,
                                    trans_a, trans_b, precision, ir_version), ie_device, precision,
            ir_version,
            temp_dir=temp_dir)


class PytorchLayerTest(CommonLayerTest):
    def produce_model_path(self, framework_model, save_path):
        path = os.path.join(save_path, 'model.onnx')
        self.torch_model = framework_model['model']
        torch.onnx.export(self.torch_model, framework_model['var'], path, input_names=['input'],
                          output_names=['output'])
        assert os.path.isfile(path), "model.onnx haven't been saved here: {}".format(save_path)
        return path

    def get_framework_results(self, inputs_dict, model_path):
        x = torch.tensor(inputs_dict['input'], dtype=torch.float32)
        return {'output': self.torch_model(x).numpy()}


class GemmModel(torch.nn.Module):
    def __init__(self, weights):
        super(GemmModel, self).__init__()
        self.weights = torch.from_numpy(weights)


class TestPytorchMM(PytorchLayerTest):
    def create_net(self, precision, shape, w_shape, output_shape, ir_version):
        """
            Pytorch net               IR net

            Input->MM->Output   =>    Input->FullyConnected

        """

        weights_const = np.random.randn(*w_shape).astype(np.float32)
        #   Create Pytorch model
        model = GemmModel(weights_const)

        if precision == 'FP16':
            weights_const = weights_const.astype(np.float16)

        #   Create reference IR net

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first

        ref_net = None

        return {'model': model, 'var': torch.randn(shape)}, ref_net

    test_data = [dict(shape=[1, 2048], w_shape=[2048, 3], output_shape=[1, 3])]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_pytorch_mm(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(precision, **params, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)
