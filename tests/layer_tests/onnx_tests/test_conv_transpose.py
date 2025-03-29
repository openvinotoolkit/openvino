# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestConvTranspose(OnnxRuntimeLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randn(*inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_conv_transpose(self, ir_version, input_shape, output_shape, kernel_shape, strides,
                              group=1,
                              dilations=None, pads=None, force_output_shape=False,
                              output_padding=None, bias=False,
                              auto_pad=None):
        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        weights = np.random.randn(*kernel_shape).astype(float)

        node_weights_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['kernel'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=weights.shape,
                vals=weights.flatten(),
            ),
        )

        conv_attrs = {
            'strides': strides,
            'group': group,
            'kernel_shape': kernel_shape[2:],  # As we have NCHW layout
        }

        if pads is not None:
            if not force_output_shape:
                conv_attrs.update({'pads': pads})
        else:
            pads = np.zeros(2 * (len(input_shape) - 2))
        _pads = np.array(pads).reshape([2, -1])
        if output_padding is not None:
            conv_attrs.update({'output_padding': output_padding})
        if dilations is not None:
            conv_attrs.update({'dilations': dilations})
        else:
            dilations = np.ones(len(input_shape) - 2)
        if force_output_shape:
            conv_attrs.update({'output_shape': output_shape[2:]})

        if auto_pad:
            conv_attrs.update({'auto_pad': auto_pad})

        nodes = [node_weights_def]
        if bias:
            bias_const = np.random.randint(-10, 10, kernel_shape[0]).astype(np.float32)

            node_bias_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['bias'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.FLOAT,
                    dims=bias_const.shape,
                    vals=bias_const.flatten(),
                ),
            )
            node_conv_transpose = onnx.helper.make_node(
                'ConvTranspose',
                inputs=['input', 'kernel', 'bias'],
                outputs=['output'],
                **conv_attrs
            )
            nodes.extend([node_bias_def, node_conv_transpose])
        else:
            node_conv_transpose = onnx.helper.make_node(
                'ConvTranspose',
                inputs=['input', 'kernel'],
                outputs=['output'],
                **conv_attrs
            )
            nodes.append(node_conv_transpose)

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            nodes,
            'test_conv_transpose_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_conv_transpose_model')

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #
        ref_net = None

        return onnx_net, ref_net

    common_tests_4D_precommit = [
        pytest.param(dict(input_shape=[1, 3, 10, 10], output_shape=[1, 3, 12, 12],
                          kernel_shape=[3, 3, 2, 2], strides=[1, 1], dilations=[2, 2]),
                     marks=pytest.mark.skip(reason="Skipped until fixed")),
        pytest.param(dict(input_shape=[1, 2, 20, 20], output_shape=[1, 2, 85, 85],
                          kernel_shape=[2, 1, 8, 8], strides=[4, 4], group=2,
                          output_padding=[1, 1]),
                     marks=pytest.mark.skip(reason="Skipped until fixed"))
    ]

    common_tests_4D = [
        dict(input_shape=[1, 3, 10, 10], output_shape=[1, 3, 11, 11],
             kernel_shape=[3, 3, 2, 2], strides=[1, 1]),
        dict(input_shape=[1, 3, 10, 10], output_shape=[1, 3, 20, 20],
             kernel_shape=[3, 3, 2, 2], strides=[2, 2]),
        dict(input_shape=[1, 3, 10, 10], output_shape=[1, 3, 12, 12],
             kernel_shape=[3, 3, 2, 2], strides=[1, 1], dilations=[2, 2]),
        dict(input_shape=[1, 3, 10, 10], output_shape=[1, 3, 21, 21],
             kernel_shape=[3, 3, 2, 2], strides=[2, 2], dilations=[2, 2]),
        dict(input_shape=[1, 2, 20, 20], output_shape=[1, 2, 85, 85],
             kernel_shape=[2, 1, 8, 8], strides=[4, 4], group=2, output_padding=[1, 1]),
    ]

    explicit_pads_tests_4D = common_tests_4D + [
        dict(input_shape=[1, 2, 20, 20], output_shape=[1, 2, 80, 80],
             kernel_shape=[2, 1, 8, 8], strides=[4, 4], group=2, pads=[2, 2, 2, 2]),
        dict(input_shape=[1, 2, 20, 20], output_shape=[1, 2, 87, 87],
             kernel_shape=[2, 1, 8, 8], strides=[4, 4], group=2, pads=[2, 2, 2, 2],
             dilations=[2, 2]),
        dict(input_shape=[1, 2, 20, 20], output_shape=[1, 2, 80, 80],
             kernel_shape=[2, 1, 8, 8], strides=[4, 4], group=2, pads=[2, 2, 2, 2],
             force_output_shape=True),
    ]

    valid_auto_pad_tests_4D = common_tests_4D + [
        dict(input_shape=[1, 2, 20, 20], output_shape=[1, 2, 84, 84],
             kernel_shape=[2, 1, 8, 8], strides=[4, 4], group=2),
        dict(input_shape=[1, 2, 20, 20], output_shape=[1, 2, 91, 91],
             kernel_shape=[2, 1, 8, 8], strides=[4, 4], group=2, dilations=[2, 2]),
        dict(input_shape=[1, 2, 20, 20], output_shape=[1, 2, 80, 80],
             kernel_shape=[2, 1, 8, 8], strides=[4, 4], group=2, force_output_shape=True),
    ]

    same_auto_pad_tests_4D = [
        dict(input_shape=[1, 3, 10, 10], output_shape=[1, 3, 10, 10],
             kernel_shape=[3, 3, 2, 2], strides=[1, 1]),
        dict(input_shape=[1, 3, 10, 10], output_shape=[1, 3, 20, 20],
             kernel_shape=[3, 3, 2, 2], strides=[2, 2]),
        dict(input_shape=[1, 2, 20, 20], output_shape=[1, 2, 80, 80],
             kernel_shape=[2, 1, 8, 8], strides=[4, 4], group=2),
        dict(input_shape=[1, 3, 10, 10], output_shape=[1, 3, 10, 10],
             kernel_shape=[3, 3, 2, 2], strides=[1, 1], dilations=[2, 2]),
        dict(input_shape=[1, 3, 10, 10], output_shape=[1, 3, 20, 20],
             kernel_shape=[3, 3, 2, 2], strides=[2, 2], dilations=[2, 2]),
        dict(input_shape=[1, 2, 20, 20], output_shape=[1, 2, 80, 80],
             kernel_shape=[2, 1, 8, 8], strides=[4, 4], group=2, dilations=[2, 2]),
        dict(input_shape=[1, 2, 20, 20], output_shape=[1, 2, 80, 80],
             kernel_shape=[2, 1, 8, 8], strides=[4, 4], group=2, force_output_shape=True),
        dict(input_shape=[1, 2, 20, 20], output_shape=[1, 2, 81, 81],
             kernel_shape=[2, 1, 8, 8], strides=[4, 4], group=2, output_padding=[1, 1]),
    ]

    @pytest.mark.parametrize("params", common_tests_4D_precommit)
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("auto_pad", ["NOTSET"])
    @pytest.mark.precommit
    def test_conv_transpose_4D_precommit(self, params, bias, ie_device, precision, ir_version,
                                         auto_pad, temp_dir):
        if ie_device == 'GPU' and 'dilations' in params:
            pytest.xfail('dilations are not supported on GPU')
        self._test(*self.create_conv_transpose(**params, ir_version=ir_version, bias=bias,
                                               auto_pad=auto_pad),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", explicit_pads_tests_4D)
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("auto_pad", ["NOTSET"])
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_conv_transpose_4D(self, params, bias, ie_device, precision, ir_version, auto_pad,
                               temp_dir):
        if ie_device == 'GPU' and 'dilations' in params:
            pytest.xfail('dilations are not supported on GPU')
        self._test(*self.create_conv_transpose(**params, ir_version=ir_version, bias=bias,
                                               auto_pad=auto_pad),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", valid_auto_pad_tests_4D)
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("auto_pad", ["VALID"])
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_conv_transpose_valid_auto_pad_4D(self, params, bias, ie_device, precision, ir_version,
                                              auto_pad, temp_dir):
        if ie_device == 'GPU' and 'dilations' in params:
            pytest.xfail('dilations are not supported on GPU')
        self._test(*self.create_conv_transpose(**params, ir_version=ir_version, bias=bias,
                                               auto_pad=auto_pad),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", same_auto_pad_tests_4D)
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER"])
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_conv_transpose_same_auto_pad_4D(self, params, bias, ie_device, precision, ir_version,
                                             auto_pad, temp_dir):
        if ie_device == 'GPU' and 'dilations' in params:
            pytest.xfail('dilations are not supported on GPU')
        self._test(*self.create_conv_transpose(**params, ir_version=ir_version, bias=bias,
                                               auto_pad=auto_pad),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
