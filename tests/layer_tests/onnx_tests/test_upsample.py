# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import os

import pytest
import torch
from common.layer_test_class import CommonLayerTest
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestUpsample(OnnxRuntimeLayerTest):
    def create_net(self, shape, mode, scales, opset, ir_version):
        """
            ONNX net                        IR net

            Input->Upsample->Output   =>    Input->Resample

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        assert opset in [7, 9]

        output_shape = shape.copy()
        output_shape[-1] = math.floor(scales[-1] * shape[-1])
        output_shape[-2] = math.floor(scales[-2] * shape[-2])
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        args = dict()
        nodes = []
        if opset == 7:
            args['scales'] = scales
        else:
            node_scales_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['scales'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.FLOAT,
                    dims=[len(scales)],
                    vals=scales,
                ),
            )
            nodes.append(node_scales_def)

        if mode:
            args['mode'] = mode
        node_def = helper.make_node(
            'Upsample',
            inputs=['input'] if opset == 7 else ['input', 'scales'],
            outputs=['output'],
            **args
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
        onnx_net = onnx_make_model(graph_def,
                                     producer_name='test_model',
                                     opset_imports=[helper.make_opsetid("", opset)])

        #   Create reference IR net
        mode_to_resample_type = {None: 'caffe.ResampleParameter.NEAREST',
                                 'nearest': 'caffe.ResampleParameter.NEAREST',
                                 'linear': 'caffe.ResampleParameter.LINEAR'}
        assert mode in mode_to_resample_type

        ref_net = None

        return onnx_net, ref_net

    test_data = [dict(shape=[1, 3, 10, 12], scales=[1., 1., 2., 2.]),
                 dict(shape=[1, 3, 10, 12], scales=[1., 1., 2.5, 2.5]),
                 dict(shape=[1, 3, 10, 12], scales=[1., 1., 2.5, 2.])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("mode", [None, 'nearest'])
    @pytest.mark.parametrize("opset", [7, 9])
    @pytest.mark.nightly
    def test_upsample_nearest(self, params, mode, opset, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, mode=mode, opset=opset, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("opset", [7, 9])
    @pytest.mark.nightly
    def test_upsample_linear(self, params, opset, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip('GREEN_SUITE')
        self._test(*self.create_net(**params, mode='linear', opset=opset, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)


class PytorchLayerTest(CommonLayerTest):
    def produce_model_path(self, framework_model, save_path):
        path = os.path.join(save_path, 'model.onnx')
        self.torch_model = framework_model['model']
        torch.onnx.export(self.torch_model, framework_model['var'], path, output_names=['output'])
        assert os.path.isfile(path), "model.onnx haven't been saved here: {}".format(save_path)
        return path

    def get_framework_results(self, inputs_dict, model_path):
        x = torch.tensor(inputs_dict['input'], dtype=torch.float32)
        return {'output': self.torch_model(x).numpy()}


class UpsampleModel(torch.nn.Module):
    def __init__(self, mode, size, scale_factor):
        super(UpsampleModel, self).__init__()
        args = dict()
        if mode:
            args['mode'] = mode
        if scale_factor:
            args['scale_factor'] = scale_factor
        elif size:
            args['size'] = size
        self.upsample = torch.nn.modules.upsampling.Upsample(**args)


class TestPytorchUpsample(PytorchLayerTest):
    def create_net(self, shape, mode, size, scale_factor, ir_version):
        """
            Pytorch net                        IR net

            Input->Upsample->Output   =>    Input->Resample

        """

        output_shape = shape.copy()
        if size:
            output_shape[2] = size[0]
            output_shape[3] = size[1]
        elif scale_factor:
            output_shape[2] = scale_factor * output_shape[2]
            output_shape[3] = scale_factor * output_shape[3]

        #   Create Pytorch model
        model = UpsampleModel(mode, size, scale_factor)

        #   Create reference IR net
        mode_to_resample_type = {None: 'caffe.ResampleParameter.NEAREST',
                                 'nearest': 'caffe.ResampleParameter.NEAREST',
                                 'bilinear': 'caffe.ResampleParameter.LINEAR'}
        assert mode in mode_to_resample_type

        ref_net = None

        return {'model': model, 'var': torch.randn(shape)}, ref_net

    test_data_precommit = [dict(shape=[1, 3, 10, 10], size=(25, 25), scale_factor=None),
                           dict(shape=[1, 3, 10, 10], size=None, scale_factor=2)]

    test_data = [dict(shape=[1, 3, 10, 10], size=(20, 20), scale_factor=None),
                 dict(shape=[1, 3, 10, 10], size=(25, 25), scale_factor=None),
                 dict(shape=[1, 3, 10, 10], size=None, scale_factor=2)]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.parametrize("mode", [None, 'nearest'])
    def test_pytorch_upsample_precommit(self, params, mode, ie_device, precision, ir_version,
                                        temp_dir):
        if ie_device == 'GPU':
            pytest.skip('Linear upsampling not supported on GPU')
        self._test(*self.create_net(**params, mode=mode, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("mode", [None, 'nearest', 'bilinear'])
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_pytorch_upsample(self, params, mode, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU' and mode == 'bilinear':
            pytest.skip('Linear upsampling not supported on GPU')
        self._test(*self.create_net(**params, mode=mode, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)
