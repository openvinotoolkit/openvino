# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestImageScaler(OnnxRuntimeLayerTest):
    def create_net(self, shape, scale, ir_version):
        """
            ONNX net                           IR net

            Input->ImageScaler->Output   =>    Input->ScaleShift(Power)

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

        bias = np.random.randint(-10, 10, shape[1]).astype(float)

        node_def = onnx.helper.make_node(
            'ImageScaler',
            inputs=['input'],
            outputs=['output'],
            bias=bias,
            scale=scale
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

        #
        #   Create reference IR net
        #

        ref_net = None

        return onnx_net, ref_net

    def create_net_const(self, shape, scale, precision, ir_version):
        """
            ONNX net                                     IR net

            Input->Concat(+scaled const)->Output   =>    Input->Concat(+const)

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        concat_axis = 0
        output_shape = shape.copy()
        output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        constant = np.random.randint(-127, 127, shape).astype(float)
        bias = np.random.randint(-10, 10, shape[1]).astype(float)

        node_const_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const1'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=constant.shape,
                vals=constant.flatten(),
            ),
        )

        node_def = onnx.helper.make_node(
            'ImageScaler',
            inputs=['const1'],
            outputs=['scale'],
            bias=bias,
            scale=scale
        )

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['input', 'scale'],
            outputs=['output'],
            axis=concat_axis
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_def, node_concat_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #
        ir_const = constant * scale + np.expand_dims(np.expand_dims([bias], 2), 3)
        if precision == 'FP16':
            ir_const = ir_const.astype(np.float16)

        ref_net = None

        return onnx_net, ref_net

    test_data_precommit = [dict(shape=[2, 4, 6, 8], scale=4.5),
                           dict(shape=[1, 1, 10, 12], scale=0.5)]

    test_data = [dict(shape=[1, 1, 10, 12], scale=0.5),
                 dict(shape=[1, 3, 10, 12], scale=1.5),
                 dict(shape=[6, 8, 10, 12], scale=4.5)]

    @pytest.mark.parametrize("params", test_data_precommit)
    def test_image_scaler_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_image_scaler(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    def test_image_scaler_const_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_image_scaler_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
