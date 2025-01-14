# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestSum(OnnxRuntimeLayerTest):
    def create_net(self, dyn_shapes, const_shapes, precision, ir_version, opset=None):
        """
            ONNX net                                IR net

            Inputs->Sum with consts->Output   =>    Input->Eltwise
        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        inputs = list()
        input_names = list()
        out_shape_len = 0
        for i, shape in enumerate(dyn_shapes):
            input_name = 'input{}'.format(i + 1)
            inputs.append(helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape))
            input_names.append(input_name)
            if len(shape) > out_shape_len:
                out_shape_len = len(shape)
                output_shape = shape
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        nodes = list()
        consts = list()
        for i, shape in enumerate(const_shapes):
            const = np.random.randint(-127, 127, shape).astype(float)
            const_name = 'const{}'.format(i + 1)
            nodes.append(helper.make_node(
                'Constant',
                inputs=[],
                outputs=[const_name],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.FLOAT,
                    dims=const.shape,
                    vals=const.flatten(),
                ),
            ))
            input_names.append(const_name)
            consts.append(const)

        nodes.append(helper.make_node(
            'Sum',
            inputs=input_names,
            outputs=['output']
        ))

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            nodes,
            'test_model',
            inputs,
            [output],
        )

        # Create the model (ModelProto)
        args = dict(producer_name='test_model')
        if opset:
            args['opset_imports'] = [helper.make_opsetid("", opset)]
        onnx_net = onnx_make_model(graph_def, **args)

        #   Create reference IR net

        ref_net = None
        # Too complicated IR to generate by hand

        return onnx_net, ref_net

    def create_const_net(self, const_shapes, ir_version, opset=None):
        """
            ONNX net                                          IR net

            Inputs->Concat with Sum of consts->Output   =>    Input->Concat with consts
        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        shape_len = 0
        for shape in const_shapes:
            if len(shape) > shape_len:
                shape_len = len(shape)
                input_shape = shape

        concat_axis = 0
        output_shape = input_shape.copy()
        output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        nodes = list()
        input_names = list()
        consts = list()
        for i, shape in enumerate(const_shapes):
            const = np.random.randint(-127, 127, shape).astype(float)
            const_name = 'const{}'.format(i + 1)
            nodes.append(helper.make_node(
                'Constant',
                inputs=[],
                outputs=[const_name],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.FLOAT,
                    dims=const.shape,
                    vals=const.flatten(),
                ),
            ))
            input_names.append(const_name)
            consts.append(const)

        nodes.append(helper.make_node(
            'Sum',
            inputs=input_names,
            outputs=['sum']
        ))

        nodes.append(helper.make_node(
            'Concat',
            inputs=['input', 'sum'],
            outputs=['output'],
            axis=concat_axis
        ))

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

        #   Create reference IR net

        ref_net = None

        return onnx_net, ref_net

    test_data_precommit = [
        dict(dyn_shapes=[[4, 6, 8], [4, 6, 8], [4, 6, 8]],
             const_shapes=[[4, 6, 8], [4, 6, 8], [4, 6, 8]]),
        dict(dyn_shapes=[[4, 6, 8, 10, 12]],
             const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12]]),
        dict(dyn_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12]],
             const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12]])]

    test_data = [
        # TODO: Add broadcasting tests. Note: Sum-6 doesn't support broadcasting
        dict(dyn_shapes=[[4, 6]], const_shapes=[[4, 6]]),
        dict(dyn_shapes=[[4, 6]], const_shapes=[[4, 6], [4, 6]]),
        dict(dyn_shapes=[[4, 6]], const_shapes=[[4, 6], [4, 6], [4, 6]]),
        dict(dyn_shapes=[[4, 6], [4, 6]], const_shapes=[]),
        dict(dyn_shapes=[[4, 6], [4, 6]], const_shapes=[[4, 6]]),
        dict(dyn_shapes=[[4, 6], [4, 6]], const_shapes=[[4, 6], [4, 6]]),
        dict(dyn_shapes=[[4, 6], [4, 6]], const_shapes=[[4, 6], [4, 6], [4, 6]]),
        dict(dyn_shapes=[[4, 6], [4, 6], [4, 6]], const_shapes=[]),
        dict(dyn_shapes=[[4, 6], [4, 6], [4, 6]], const_shapes=[[4, 6]]),
        dict(dyn_shapes=[[4, 6], [4, 6], [4, 6]], const_shapes=[[4, 6], [4, 6]]),
        dict(dyn_shapes=[[4, 6], [4, 6], [4, 6]], const_shapes=[[4, 6], [4, 6], [4, 6]]),
        dict(dyn_shapes=[[4, 6, 8]], const_shapes=[[4, 6, 8]]),
        dict(dyn_shapes=[[4, 6, 8]], const_shapes=[[4, 6, 8], [4, 6, 8]]),
        dict(dyn_shapes=[[4, 6, 8]], const_shapes=[[4, 6, 8], [4, 6, 8], [4, 6, 8]]),
        dict(dyn_shapes=[[4, 6, 8], [4, 6, 8]], const_shapes=[]),
        dict(dyn_shapes=[[4, 6, 8], [4, 6, 8]], const_shapes=[[4, 6, 8]]),
        dict(dyn_shapes=[[4, 6, 8], [4, 6, 8]], const_shapes=[[4, 6, 8], [4, 6, 8]]),
        dict(dyn_shapes=[[4, 6, 8], [4, 6, 8]], const_shapes=[[4, 6, 8], [4, 6, 8], [4, 6, 8]]),
        dict(dyn_shapes=[[4, 6, 8], [4, 6, 8], [4, 6, 8]], const_shapes=[]),
        dict(dyn_shapes=[[4, 6, 8], [4, 6, 8], [4, 6, 8]], const_shapes=[[4, 6, 8]]),
        dict(dyn_shapes=[[4, 6, 8], [4, 6, 8], [4, 6, 8]], const_shapes=[[4, 6, 8], [4, 6, 8]]),
        dict(dyn_shapes=[[4, 6, 8], [4, 6, 8], [4, 6, 8]],
             const_shapes=[[4, 6, 8], [4, 6, 8], [4, 6, 8]]),
        dict(dyn_shapes=[[4, 6, 8, 10]], const_shapes=[[4, 6, 8, 10]]),
        dict(dyn_shapes=[[4, 6, 8, 10]], const_shapes=[[4, 6, 8, 10], [4, 6, 8, 10]]),
        dict(dyn_shapes=[[4, 6, 8, 10]],
             const_shapes=[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]]),
        dict(dyn_shapes=[[4, 6, 8, 10], [4, 6, 8, 10]], const_shapes=[]),
        dict(dyn_shapes=[[4, 6, 8, 10], [4, 6, 8, 10]], const_shapes=[[4, 6, 8, 10]]),
        dict(dyn_shapes=[[4, 6, 8, 10], [4, 6, 8, 10]],
             const_shapes=[[4, 6, 8, 10], [4, 6, 8, 10]]),
        dict(dyn_shapes=[[4, 6, 8, 10], [4, 6, 8, 10]],
             const_shapes=[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]]),
        dict(dyn_shapes=[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]], const_shapes=[]),
        dict(dyn_shapes=[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]],
             const_shapes=[[4, 6, 8, 10]]),
        dict(dyn_shapes=[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]],
             const_shapes=[[4, 6, 8, 10], [4, 6, 8, 10]]),
        dict(dyn_shapes=[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]],
             const_shapes=[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]]),
        dict(dyn_shapes=[[4, 6, 8, 10, 12]], const_shapes=[[4, 6, 8, 10, 12]]),
        dict(dyn_shapes=[[4, 6, 8, 10, 12]], const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12]]),
        dict(dyn_shapes=[[4, 6, 8, 10, 12]],
             const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12]]),
        dict(dyn_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12]], const_shapes=[]),
        dict(dyn_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12]], const_shapes=[[4, 6, 8, 10, 12]]),
        dict(dyn_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12]],
             const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12]]),
        dict(dyn_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12]],
             const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12]]),
        dict(dyn_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12]], const_shapes=[]),
        dict(dyn_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12]],
             const_shapes=[[4, 6, 8, 10, 12]]),
        dict(dyn_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12]],
             const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12]]),
        dict(dyn_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12]],
             const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12]])]

    const_test_data_precommit = [
        dict(const_shapes=[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]]),
        dict(const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12],
                           [4, 6, 8, 10, 12]])
    ]

    const_test_data = [
        dict(const_shapes=[[4, 6], [4, 6]]),
        dict(const_shapes=[[4, 6], [4, 6], [4, 6]]),
        dict(const_shapes=[[4, 6], [4, 6], [4, 6], [4, 6]]),
        dict(const_shapes=[[4, 6, 8], [4, 6, 8]]),
        dict(const_shapes=[[4, 6, 8], [4, 6, 8], [4, 6, 8]]),
        dict(const_shapes=[[4, 6, 8], [4, 6, 8], [4, 6, 8], [4, 6, 8]]),
        dict(const_shapes=[[4, 6, 8, 10], [4, 6, 8, 10]]),
        dict(const_shapes=[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]]),
        dict(const_shapes=[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]]),
        dict(const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12]]),
        dict(const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12]]),
        dict(const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12],
                           [4, 6, 8, 10, 12]])
    ]

    const_test_data_broadcasting_precommit = [
        dict(const_shapes=[[4, 6, 8, 10], [10], [10], [10]]),
        dict(const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [12]]),
        dict(const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [12]])
    ]

    const_test_data_broadcasting = [
        dict(const_shapes=[[4, 6], [6]]),
        dict(const_shapes=[[4, 6], [6], [6]]),
        dict(const_shapes=[[4, 6], [4, 6], [6]]),
        dict(const_shapes=[[4, 6], [6], [6], [6]]),
        dict(const_shapes=[[4, 6], [4, 6], [6], [6]]),
        dict(const_shapes=[[4, 6], [4, 6], [4, 6], [6]]),
        dict(const_shapes=[[4, 6, 8], [8]]),
        dict(const_shapes=[[4, 6, 8], [8], [8]]),
        dict(const_shapes=[[4, 6, 8], [4, 6, 8], [8]]),
        dict(const_shapes=[[4, 6, 8], [8], [8], [8]]),
        dict(const_shapes=[[4, 6, 8], [4, 6, 8], [8], [8]]),
        dict(const_shapes=[[4, 6, 8], [4, 6, 8], [4, 6, 8], [8]]),
        dict(const_shapes=[[4, 6, 8, 10], [10]]),
        dict(const_shapes=[[4, 6, 8, 10], [10], [10]]),
        dict(const_shapes=[[4, 6, 8, 10], [4, 6, 8, 10], [10]]),
        dict(const_shapes=[[4, 6, 8, 10], [10], [10], [10]]),
        dict(const_shapes=[[4, 6, 8, 10], [4, 6, 8, 10], [10], [10]]),
        dict(const_shapes=[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10], [10]]),
        dict(const_shapes=[[4, 6, 8, 10, 12], [12]]),
        dict(const_shapes=[[4, 6, 8, 10, 12], [12], [12]]),
        dict(const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [12]]),
        dict(const_shapes=[[4, 6, 8, 10, 12], [12], [12], [12]]),
        dict(const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [12], [12]]),
        dict(const_shapes=[[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [12]])
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_sum_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, precision=precision, opset=6, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_sum_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_sum(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net(**params, precision=precision, ir_version=ir_version), ie_device,
            precision, ir_version,
            temp_dir=temp_dir)

    @pytest.mark.parametrize("params", const_test_data)
    @pytest.mark.nightly
    def test_sum_const_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_const_net(**params, opset=6, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", const_test_data_precommit)
    @pytest.mark.precommit
    def test_sum_const_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_const_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", const_test_data)
    @pytest.mark.nightly
    def test_sum_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_const_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", const_test_data_broadcasting_precommit)
    @pytest.mark.precommit
    def test_sum_const_broadcasting_precommit(self, params, ie_device, precision, ir_version,
                                              temp_dir):
        self._test(*self.create_const_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", const_test_data_broadcasting)
    @pytest.mark.nightly
    def test_sum_const_broadcasting(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_const_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)
