# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestUnsqueeze(OnnxRuntimeLayerTest):
    def create_unsqueeze_net(self, axes, input_shape, output_shape, ir_version):
        """
            ONNX net                                  IR net

            Input->Unsqueeze(axes=0)->Output   =>    Input->Reshape

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        node_squeeze_def = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['input'],
            outputs=['output'],
            axes=axes
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_squeeze_def],
            'test_squeeze_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_squeeze_model')

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return onnx_net, ref_net

    def create_unsqueeze_net_const(self, axes, input_shape, output_shape, ir_version):
        """
            ONNX net                                         IR net

            Input->Concat(+unsqueezed const)->Output   =>    Input->Concat(+const)

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto
        import numpy as np

        concat_axis = 1
        concat_output_shape = output_shape.copy()
        concat_output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, output_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, concat_output_shape)

        const_number = np.prod(input_shape)
        constant = np.random.randint(-127, 127, const_number).astype(float)
        constant = np.reshape(constant, input_shape)

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

        node_squeeze_def = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['const1'],
            outputs=['unsqueeze1'],
            axes=axes
        )

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['input', 'unsqueeze1'],
            outputs=['output'],
            axis=concat_axis
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_squeeze_def, node_concat_def],
            'test_unsqueeze_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_unsqueeze_model')

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #
        ref_net = None

        return onnx_net, ref_net

    test_data_5D = [
        dict(axes=[0], input_shape=[2, 3, 10, 10], output_shape=[1, 2, 3, 10, 10]),
        dict(axes=[1], input_shape=[2, 3, 10, 10], output_shape=[2, 1, 3, 10, 10]),
        dict(axes=[2], input_shape=[2, 3, 10, 10], output_shape=[2, 3, 1, 10, 10]),
        dict(axes=[3], input_shape=[2, 3, 10, 10], output_shape=[2, 3, 10, 1, 10]),
        dict(axes=[4], input_shape=[2, 3, 10, 10], output_shape=[2, 3, 10, 10, 1]),
        dict(axes=[0, 1], input_shape=[3, 10, 10], output_shape=[1, 1, 3, 10, 10]),
        dict(axes=[0, 2], input_shape=[3, 10, 10], output_shape=[1, 3, 1, 10, 10]),
        dict(axes=[0, 3], input_shape=[3, 10, 10], output_shape=[1, 3, 10, 1, 10]),
        dict(axes=[0, 4], input_shape=[3, 10, 10], output_shape=[1, 3, 10, 10, 1]),
        dict(axes=[1, 2], input_shape=[3, 10, 10], output_shape=[3, 1, 1, 10, 10]),
        dict(axes=[1, 3], input_shape=[3, 10, 10], output_shape=[3, 1, 10, 1, 10]),
        dict(axes=[1, 4], input_shape=[3, 10, 10], output_shape=[3, 1, 10, 10, 1]),
        dict(axes=[2, 3], input_shape=[3, 10, 10], output_shape=[3, 10, 1, 1, 10]),
        dict(axes=[2, 4], input_shape=[3, 10, 10], output_shape=[3, 10, 1, 10, 1]),
        dict(axes=[3, 4], input_shape=[3, 10, 10], output_shape=[3, 10, 10, 1, 1]),
        dict(axes=[0, 1, 2], input_shape=[10, 10], output_shape=[1, 1, 1, 10, 10]),
        dict(axes=[0, 1, 3], input_shape=[10, 10], output_shape=[1, 1, 10, 1, 10]),
        dict(axes=[0, 1, 4], input_shape=[10, 10], output_shape=[1, 1, 10, 10, 1]),
        dict(axes=[0, 2, 3], input_shape=[10, 10], output_shape=[1, 10, 1, 1, 10]),
        dict(axes=[0, 2, 4], input_shape=[10, 10], output_shape=[1, 10, 1, 10, 1]),
        dict(axes=[0, 3, 4], input_shape=[10, 10], output_shape=[1, 10, 10, 1, 1]),
        dict(axes=[1, 2, 3], input_shape=[10, 10], output_shape=[10, 1, 1, 1, 10]),
        dict(axes=[1, 2, 4], input_shape=[10, 10], output_shape=[10, 1, 1, 10, 1]),
        dict(axes=[1, 3, 4], input_shape=[10, 10], output_shape=[10, 1, 10, 1, 1]),
        dict(axes=[2, 3, 4], input_shape=[10, 10], output_shape=[10, 10, 1, 1, 1])]

    test_data_4D = [
        dict(axes=[0], input_shape=[3, 10, 10], output_shape=[1, 3, 10, 10]),
        dict(axes=[1], input_shape=[3, 10, 10], output_shape=[3, 1, 10, 10]),
        dict(axes=[2], input_shape=[3, 10, 10], output_shape=[3, 10, 1, 10]),
        dict(axes=[3], input_shape=[3, 10, 10], output_shape=[3, 10, 10, 1]),
        dict(axes=[3], input_shape=[3, 10, 10], output_shape=[3, 10, 10, 1]),
        dict(axes=[0, 1], input_shape=[10, 10], output_shape=[1, 1, 10, 10]),
        dict(axes=[0, 2], input_shape=[10, 10], output_shape=[1, 10, 1, 10]),
        dict(axes=[0, 3], input_shape=[10, 10], output_shape=[1, 10, 10, 1]),
        dict(axes=[1, 2], input_shape=[10, 10], output_shape=[10, 1, 1, 10]),
        dict(axes=[1, 3], input_shape=[10, 10], output_shape=[10, 1, 10, 1]),
        dict(axes=[2, 3], input_shape=[10, 10], output_shape=[10, 10, 1, 1])]

    test_data_3D = [
        dict(axes=[0], input_shape=[10, 10], output_shape=[1, 10, 10]),
        dict(axes=[1], input_shape=[10, 10], output_shape=[10, 1, 10]),
        dict(axes=[2], input_shape=[10, 10], output_shape=[10, 10, 1])]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_unsqueeze_5D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_unsqueeze_net(**params, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_unsqueeze_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_unsqueeze_net(**params, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_unsqueeze_3D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_unsqueeze_net(**params, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_unsqueeze_const_5D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_unsqueeze_net_const(**params, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_unsqueeze_const_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_unsqueeze_net_const(**params, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_unsqueeze_const_3D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_unsqueeze_net_const(**params, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)
