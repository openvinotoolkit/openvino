# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestOperations(OnnxRuntimeLayerTest):
    def create_net(self, shape1, shape2, op, precision, ir_version, opset=None):
        """
            ONNX net                                  IR net

            Input->Add/Mul with const->Output   =>    Input->Eltwise
        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        if op not in ['Add', 'Sub', 'Mul', 'Div']:
            raise ValueError("Operation has to be either Add or Mul or Sub or Div")

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape1)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape1)

        min_val = 1 if op == 'Div' else -127
        if shape2:
            const = np.random.randint(min_val, 127, shape2).astype(float)
        else:
            const = np.random.randint(min_val, 127, 1).astype(float)
            # TODO: add check when MO remove redundant layer (as Add/Sub if const = 0 or Mul/Div if const = 1)
            if const in [0, 1]:
                const = np.array([2], dtype=float)

        node_const_def = helper.make_node(
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

        node_def = helper.make_node(
            op,
            inputs=['input', 'const'],
            outputs=['output']
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_def],
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
        if op == 'Div':
            const = np.power(const, -1)
        elif op == 'Sub':
            const = -const

        ref_net = None

        return onnx_net, ref_net

    def create_net_const(self, shape1, shape2, op, precision, ir_version, opset=None):
        """
            ONNX net                                                      IR net

            Input->Concat with two added/multiplied consts->Output   =>   Input->Concat
        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        if op not in ['Add', 'Sub', 'Mul', 'Div']:
            raise ValueError("op has to be either Add or Mul")

        concat_axis = 0
        output_shape = list(shape1)
        output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape1)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        const1 = np.random.randint(-127, 127, shape1).astype(float)
        min_val = 1 if op == 'Div' else -127
        if shape2:
            const2 = np.random.randint(min_val, 127, shape2).astype(float)
        else:
            const2 = np.random.randint(min_val, 127, 1).astype(float)

        node_const1_def = helper.make_node(
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

        node_const2_def = helper.make_node(
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

        node_def = helper.make_node(
            op,
            inputs=['const1', 'const2'],
            outputs=['node_out']
        )

        node_concat_def = helper.make_node(
            'Concat',
            inputs=['input', 'node_out'],
            outputs=['output'],
            axis=concat_axis
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const1_def, node_const2_def, node_def, node_concat_def],
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
        if op == 'Add':
            constant_calculated = np.add(const1, const2)
        elif op == 'Sub':
            constant_calculated = np.subtract(const1, const2)
        elif op == 'Mul':
            constant_calculated = np.multiply(const1, const2)
        elif op == 'Div':
            constant_calculated = np.divide(const1, const2)

        if precision == 'FP16':
            constant_calculated = constant_calculated.astype(np.float16)

        ref_net = None

        return onnx_net, ref_net

    test_data_precommit = [dict(shape1=[2, 4], shape2=[2, 4]),
                           # scalar cases
                           dict(shape1=[2, 4], shape2=None)]

    test_data = [dict(shape1=[4, 6], shape2=[4, 6]),
                 dict(shape1=[4, 6, 8], shape2=[4, 6, 8]),
                 dict(shape1=[4, 6, 8, 10], shape2=[4, 6, 8, 10]),
                 dict(shape1=[4, 6, 8, 10, 12], shape2=[4, 6, 8, 10, 12]),
                 # scalar cases
                 dict(shape1=[4, 6], shape2=None),
                 dict(shape1=[4, 6, 8], shape2=None),
                 dict(shape1=[4, 6, 8, 10], shape2=None),
                 dict(shape1=[4, 6, 8, 10, 12], shape2=None)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_add(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op='Add', precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_add_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, op='Add', precision=precision, ir_version=ir_version),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_sub(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op='Sub', precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_sub_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, op='Sub', precision=precision, ir_version=ir_version),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_mul(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op='Mul', precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_mul_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, op='Mul', precision=precision, ir_version=ir_version),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_div(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op='Div', precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_div_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, op='Div', precision=precision, ir_version=ir_version),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_add_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op='Add', precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_add_const_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, op='Add', precision=precision, ir_version=ir_version),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_sub_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op='Sub', precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_sub_const_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, op='Sub', precision=precision, ir_version=ir_version),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_mul_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op='Mul', precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_mul_const_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, op='Mul', precision=precision, ir_version=ir_version),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_div_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op='Div', precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_div_const_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, op='Div', precision=precision, ir_version=ir_version),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_add_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op='Add', precision=precision, opset=6,
                                    ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_add_const_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, op='Add', precision=precision, opset=6,
                                          ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_sub_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op='Sub', precision=precision, opset=6,
                                    ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_sub_const_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, op='Sub', precision=precision, opset=6,
                                          ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_mul_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op='Mul', precision=precision, opset=6,
                                    ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_mul_const_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, op='Mul', precision=precision, opset=6,
                                          ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_div_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op='Div', precision=precision, opset=6,
                                    ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_div_const_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, op='Div', precision=precision, opset=6,
                                          ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
