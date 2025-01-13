# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestFlatten(OnnxRuntimeLayerTest):
    def create_flatten_net(self, axis, input_shape, dim, ir_version, opset=None):
        """
            ONNX net                       IR net

            Input->Flatten->Output   =>    Input->Reshape

        """

        #
        #   Create ONNX model
        #

        # TODO: possible move all imports to separate func?
        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, dim)

        node_flatten_def = onnx.helper.make_node(
            'Flatten',
            inputs=['input'],
            outputs=['output'],
            axis=axis,
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_flatten_def],
            'test_flatten_model',
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
        #   Please, spesify 'type': 'Input' for inpit node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return onnx_net, ref_net

    def create_flatten_net_const(self, axis, input_shape, dim, ir_version, opset=None):
        """
            ONNX net                               IR net

            Input->Flatten->Concat->Output   =>    Input->Concat
                     Input-'                       Const-'

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto
        import numpy as np

        concat_axis = 0
        concat_output_shape = dim.copy()
        concat_output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, dim)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, concat_output_shape)

        const_number = np.prod(input_shape)
        constant = np.random.randint(-127, 127, const_number).astype(float)

        node_const_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=input_shape,
                vals=constant,
            ),
        )

        node_flatten_def = onnx.helper.make_node(
            'Flatten',
            inputs=['const'],
            outputs=['flatten_output'],
            axis=axis,
        )

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['input', 'flatten_output'],
            outputs=['output'],
            axis=concat_axis
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_flatten_def, node_concat_def],
            'test_flatten_model',
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
        #   Please, spesify 'type': 'Input' for inpit node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return onnx_net, ref_net

    test_data_3D = [
        dict(axis=0, input_shape=[1, 3, 224], dim=[1, 672]),
        dict(axis=-3, input_shape=[1, 3, 224], dim=[1, 672]),
        dict(axis=1, input_shape=[1, 3, 224], dim=[1, 672]),
        dict(axis=-2, input_shape=[1, 3, 224], dim=[1, 672]),
        dict(axis=2, input_shape=[2, 3, 224], dim=[6, 224]),
        dict(axis=-1, input_shape=[2, 3, 224], dim=[6, 224]),
        dict(axis=3, input_shape=[3, 3, 224], dim=[2016, 1])
    ]

    test_data_4D_precommit = [
        dict(axis=1, input_shape=[1, 3, 224, 224], dim=[1, 150528]),
        dict(axis=-3, input_shape=[1, 3, 224, 224], dim=[1, 150528])
    ]

    test_data_4D = [
        dict(axis=0, input_shape=[1, 3, 224, 224], dim=[1, 150528]),
        dict(axis=-4, input_shape=[1, 3, 224, 224], dim=[1, 150528]),
        dict(axis=1, input_shape=[1, 3, 224, 224], dim=[1, 150528]),
        dict(axis=-3, input_shape=[1, 3, 224, 224], dim=[1, 150528]),
        dict(axis=2, input_shape=[2, 3, 224, 224], dim=[6, 50176]),
        dict(axis=-2, input_shape=[2, 3, 224, 224], dim=[6, 50176]),
        dict(axis=3, input_shape=[3, 3, 224, 224], dim=[2016, 224]),
        dict(axis=-1, input_shape=[3, 3, 224, 224], dim=[2016, 224]),
        dict(axis=4, input_shape=[4, 3, 224, 224], dim=[602112, 1])
    ]

    test_data_5D_precommit = [
        dict(axis=-5, input_shape=[1, 3, 9, 224, 224], dim=[1, 1354752]),
        dict(axis=5, input_shape=[4, 3, 9, 224, 224], dim=[5419008, 1])]

    test_data_5D = [
        dict(axis=0, input_shape=[1, 3, 9, 224, 224], dim=[1, 1354752]),
        dict(axis=-5, input_shape=[1, 3, 9, 224, 224], dim=[1, 1354752]),
        dict(axis=1, input_shape=[1, 3, 9, 224, 224], dim=[1, 1354752]),
        dict(axis=-4, input_shape=[1, 3, 9, 224, 224], dim=[1, 1354752]),
        dict(axis=2, input_shape=[2, 3, 9, 224, 224], dim=[6, 451584]),
        dict(axis=-3, input_shape=[2, 3, 9, 224, 224], dim=[6, 451584]),
        dict(axis=3, input_shape=[3, 3, 9, 224, 224], dim=[81, 50176]),
        dict(axis=-2, input_shape=[3, 3, 9, 224, 224], dim=[81, 50176]),
        dict(axis=4, input_shape=[3, 3, 9, 224, 224], dim=[18144, 224]),
        dict(axis=-1, input_shape=[3, 3, 9, 224, 224], dim=[18144, 224]),
        dict(axis=5, input_shape=[4, 3, 9, 224, 224], dim=[5419008, 1])
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.parametrize("opset", [6, 9])
    @pytest.mark.nightly
    def test_flatten_3D(self, params, opset, ie_device, precision, ir_version, temp_dir):
        # negative axis not allowed by onnx spec for flatten-1 and flatten-9
        if params['axis'] < 0:
            self.skip_framework = True
        else:
            self.skip_framework = False
        self._test(*self.create_flatten_net(**params, ir_version=ir_version, opset=opset),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.parametrize("opset", [6, 9])
    @pytest.mark.nightly
    def test_flatten_3D_const(self, params, opset, ie_device, precision, ir_version, temp_dir):
        # negative axis not allowed by onnx spec for flatten-1 and flatten-9
        if params['axis'] < 0:
            self.skip_framework = True
        else:
            self.skip_framework = False
        self._test(*self.create_flatten_net_const(**params, ir_version=ir_version, opset=opset),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.parametrize("opset", [6, 9])
    @pytest.mark.nightly
    def test_flatten_4D(self, params, opset, ie_device, precision, ir_version, temp_dir):
        # negative axis not allowed by onnx spec for flatten-1 and flatten-9
        if params['axis'] < 0:
            self.skip_framework = True
        else:
            self.skip_framework = False
        self._test(*self.create_flatten_net(**params, ir_version=ir_version, opset=opset),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D_precommit)
    @pytest.mark.parametrize("opset", [6, 9])
    @pytest.mark.precommit
    def test_flatten_4D_precommit(self, params, opset, ie_device, precision, ir_version, temp_dir):
        # negative axis not allowed by onnx spec for flatten-1 and flatten-9
        if params['axis'] < 0:
            self.skip_framework = True
        else:
            self.skip_framework = False
        self._test(*self.create_flatten_net(**params, ir_version=ir_version, opset=opset),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D_precommit)
    @pytest.mark.parametrize("opset", [6, 9])
    @pytest.mark.nightly
    def test_flatten_4D_const_precommit(self, params, opset, ie_device, precision, ir_version,
                                        temp_dir):
        # negative axis not allowed by onnx spec for flatten-1 and flatten-9
        if params['axis'] < 0:
            self.skip_framework = True
        else:
            self.skip_framework = False
        self._test(*self.create_flatten_net_const(**params, ir_version=ir_version, opset=opset),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.parametrize("opset", [6, 9])
    @pytest.mark.nightly
    def test_flatten_4D_const(self, params, opset, ie_device, precision, ir_version, temp_dir):
        # negative axis not allowed by onnx spec for flatten-1 and flatten-9
        if params['axis'] < 0:
            self.skip_framework = True
        else:
            self.skip_framework = False
        self._test(*self.create_flatten_net_const(**params, ir_version=ir_version, opset=opset),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D_precommit)
    @pytest.mark.parametrize("opset", [6, 9])
    @pytest.mark.nightly
    def test_flatten_5D_precommit(self, params, opset, ie_device, precision, ir_version, temp_dir):
        # negative axis not allowed by onnx spec for flatten-1 and flatten-9
        if params['axis'] < 0:
            self.skip_framework = True
        else:
            self.skip_framework = False
        self._test(*self.create_flatten_net(**params, ir_version=ir_version, opset=opset),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.parametrize("opset", [6, 9])
    @pytest.mark.nightly
    def test_flatten_5D(self, params, opset, ie_device, precision, ir_version, temp_dir):
        # negative axis not allowed by onnx spec for flatten-1 and flatten-9
        if params['axis'] < 0:
            self.skip_framework = True
        else:
            self.skip_framework = False
        self._test(*self.create_flatten_net(**params, ir_version=ir_version, opset=opset),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D_precommit)
    @pytest.mark.parametrize("opset", [6, 9])
    @pytest.mark.nightly
    def test_flatten_5D_const_precommit(self, params, opset, ie_device, precision, ir_version,
                                        temp_dir):
        # negative axis not allowed by onnx spec for flatten-1 and flatten-9
        if params['axis'] < 0:
            self.skip_framework = True
        else:
            self.skip_framework = False
        self._test(*self.create_flatten_net_const(**params, ir_version=ir_version, opset=opset),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.parametrize("opset", [6, 9])
    @pytest.mark.nightly
    def test_flatten_5D_const(self, params, opset, ie_device, precision, ir_version, temp_dir):
        # negative axis not allowed by onnx spec for flatten-1 and flatten-9
        if params['axis'] < 0:
            self.skip_framework = True
        else:
            self.skip_framework = False
        self._test(*self.create_flatten_net_const(**params, ir_version=ir_version, opset=opset),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
