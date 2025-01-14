# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestSlice(OnnxRuntimeLayerTest):
    def create_net(self, shape, axes, ends, starts, ir_version, opset=6, steps=None):
        """
            ONNX net                    IR net

            Input->Slice->Output   =>    Input->Crop

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        # calculate output shape
        test_arr = np.zeros(shape)
        slice_idx = [None] * len(shape)
        for i, axis in enumerate(axes):
            slice_idx[axis] = slice(starts[i], ends[i], steps[i] if steps is not None else 1)
        for axis, s in enumerate(slice_idx):
            if s is None:
                slice_idx[axis] = slice(0, shape[axis], 1)
        test_arr = test_arr[tuple(slice_idx)]

        output_shape = list(test_arr.shape)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        nodes = list()
        if opset < 10:
            node_def = onnx.helper.make_node(
                'Slice',
                inputs=['input'],
                outputs=['slice'],
                starts=starts,
                ends=ends,
                axes=axes
            )
            nodes.append(node_def)
        else:
            node_starts_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['starts'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.INT64,
                    dims=[len(starts)],
                    vals=starts
                )
            )
            node_ends_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['ends'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.INT64,
                    dims=[len(ends)],
                    vals=ends
                )
            )
            node_axes_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['axes'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.INT64,
                    dims=[len(axes)],
                    vals=axes
                )
            )
            inputs = ['input', 'starts', 'ends', 'axes']
            if steps:
                node_steps_def = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['steps'],
                    value=helper.make_tensor(
                        name='const_tensor',
                        data_type=TensorProto.INT64,
                        dims=[len(steps)],
                        vals=steps
                    )
                )
                nodes.append(node_steps_def)
                inputs.append('steps')

            node_def = onnx.helper.make_node(
                'Slice',
                inputs=inputs,
                outputs=['slice']
            )
            nodes.extend([node_starts_def, node_ends_def, node_axes_def, node_def])

        elu_def = onnx.helper.make_node(
            'Elu',
            inputs=['slice'],
            outputs=['output']
        )
        nodes.append(elu_def)

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            nodes,
            'test_model',
            [input],
            [output]
        )

        # Create the model (ModelProto)
        args = dict(producer_name='test_model')
        if opset:
            args['opset_imports'] = [helper.make_opsetid("", opset)]
        onnx_net = onnx_make_model(graph_def, **args)

        #
        #   Create reference IR net
        #

        ref_net = None

        return onnx_net, ref_net

    def create_net_const(self, shape, axes, ends, starts, ir_version, opset=6, steps=None):
        """
            ONNX net                                         IR net

            Input->Concat(+sliced const)->Output   =>    Input->Concat(+const)

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        # calculate output shape
        constant = np.random.randint(-127, 127, shape).astype(float)

        slice_idx = [None] * len(shape)
        for i, axis in enumerate(axes):
            slice_idx[axis] = slice(starts[i], ends[i], steps[i] if steps is not None else 1)

        for axis, s in enumerate(slice_idx):
            if s is None:
                slice_idx[axis] = slice(0, shape[axis], 1)

        constant_after = constant[tuple(slice_idx)]

        output_shape = list(constant_after.shape)

        concat_axis = 0
        concat_output_shape = output_shape.copy()
        concat_output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, output_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, concat_output_shape)

        node_const_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const1'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=shape,
                vals=constant.flatten(),
            ),
        )

        nodes = [node_const_def]
        if opset < 10:
            node_def = onnx.helper.make_node(
                'Slice',
                inputs=['const1'],
                outputs=['slice'],
                starts=starts,
                ends=ends,
                axes=axes
            )
            nodes.append(node_def)
        else:
            node_starts_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['starts'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.INT64,
                    dims=[len(starts)],
                    vals=starts
                )
            )
            node_ends_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['ends'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.INT64,
                    dims=[len(ends)],
                    vals=ends
                )
            )
            node_axes_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['axes'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.INT64,
                    dims=[len(axes)],
                    vals=axes
                )
            )

            inputs = ['const1', 'starts', 'ends', 'axes']
            if steps:
                node_steps_def = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['steps'],
                    value=helper.make_tensor(
                        name='const_tensor',
                        data_type=TensorProto.INT64,
                        dims=[len(steps)],
                        vals=steps
                    )
                )
                nodes.append(node_steps_def)
                inputs.append('steps')

            node_def = onnx.helper.make_node(
                'Slice',
                inputs=inputs,
                outputs=['slice']
            )
            nodes.extend([node_starts_def, node_ends_def, node_axes_def, node_def])

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['input', 'slice'],
            outputs=['output'],
            axis=concat_axis
        )
        nodes.append(node_concat_def)

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            nodes,
            'test_reshape_model',
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

        ref_net = None

        return onnx_net, ref_net

    test_data_no_steps = [
        dict(shape=[10, 12], axes=[0], starts=[1], ends=[9]),
        dict(shape=[10, 12], axes=[1], starts=[1], ends=[11]),
        dict(shape=[10, 12], axes=[0, 1], starts=[1, 1], ends=[9, 11]),
        dict(shape=[8, 10, 12], axes=[0], starts=[1], ends=[7]),
        dict(shape=[8, 10, 12], axes=[1], starts=[1], ends=[9]),
        dict(shape=[8, 10, 12], axes=[2], starts=[1], ends=[11]),
        dict(shape=[8, 10, 12], axes=[0, 1], starts=[1, 1], ends=[7, 9]),
        dict(shape=[8, 10, 12], axes=[1, 2], starts=[1, 1], ends=[9, 11]),
        dict(shape=[8, 10, 12], axes=[0, 2], starts=[1, 1], ends=[7, 11]),
        dict(shape=[8, 10, 12], axes=[0, 1, 2], starts=[1, 1, 1], ends=[7, 9, 11]),
        dict(shape=[6, 8, 10, 12], axes=[0], starts=[1], ends=[5]),
        dict(shape=[6, 8, 10, 12], axes=[1], starts=[1], ends=[7]),
        dict(shape=[6, 8, 10, 12], axes=[2], starts=[1], ends=[9]),
        dict(shape=[6, 8, 10, 12], axes=[3], starts=[1], ends=[11]),
        dict(shape=[6, 8, 10, 12], axes=[0, 1], starts=[1, 1], ends=[5, 7]),
        dict(shape=[6, 8, 10, 12], axes=[1, 2], starts=[1, 1], ends=[7, 9]),
        dict(shape=[6, 8, 10, 12], axes=[2, 3], starts=[1, 1], ends=[9, 11]),
        dict(shape=[6, 8, 10, 12], axes=[0, 2], starts=[1, 1], ends=[5, 9]),
        dict(shape=[6, 8, 10, 12], axes=[0, 3], starts=[1, 1], ends=[5, 11]),
        dict(shape=[6, 8, 10, 12], axes=[1, 3], starts=[1, 1], ends=[7, 11]),
        dict(shape=[6, 8, 10, 12], axes=[0, 1, 2], starts=[1, 1, 1], ends=[5, 7, 9]),
        dict(shape=[6, 8, 10, 12], axes=[1, 2, 3], starts=[1, 1, 1], ends=[7, 9, 11]),
        dict(shape=[6, 8, 10, 12], axes=[0, 2, 3], starts=[1, 1, 1], ends=[5, 9, 11]),
        dict(shape=[6, 8, 10, 12], axes=[0, 1, 3], starts=[1, 1, 1], ends=[5, 7, 11]),
        dict(shape=[6, 8, 10, 12], axes=[0, 1, 2, 3], starts=[1, 1, 1, 1], ends=[5, 7, 9, 11]),
        dict(shape=[4, 6, 8, 10, 12], axes=[0], starts=[1], ends=[3]),
        dict(shape=[4, 6, 8, 10, 12], axes=[1], starts=[1], ends=[5]),
        dict(shape=[4, 6, 8, 10, 12], axes=[2], starts=[1], ends=[7]),
        dict(shape=[4, 6, 8, 10, 12], axes=[3], starts=[1], ends=[9]),
        dict(shape=[4, 6, 8, 10, 12], axes=[4], starts=[1], ends=[11]),
        dict(shape=[4, 6, 8, 10, 12], axes=[0, 1], starts=[1, 1], ends=[3, 5]),
        dict(shape=[4, 6, 8, 10, 12], axes=[2, 3], starts=[1, 1], ends=[7, 9]),
        dict(shape=[4, 6, 8, 10, 12], axes=[3, 4], starts=[1, 1], ends=[9, 11]),
        dict(shape=[4, 6, 8, 10, 12], axes=[0, 1, 2], starts=[1, 1, 1], ends=[3, 5, 7]),
        dict(shape=[4, 6, 8, 10, 12], axes=[1, 2, 3], starts=[1, 1, 1], ends=[5, 7, 9]),
        dict(shape=[4, 6, 8, 10, 12], axes=[2, 3, 4], starts=[1, 1, 1], ends=[7, 9, 11]),
        dict(shape=[4, 6, 8, 10, 12], axes=[0, 1, 2, 3], starts=[1, 1, 1, 1], ends=[3, 5, 7, 9]),
        dict(shape=[4, 6, 8, 10, 12], axes=[1, 2, 3, 4], starts=[1, 1, 1, 1], ends=[5, 7, 9, 11]),
        dict(shape=[4, 6, 8, 10, 12], axes=[0, 1, 2, 3, 4], starts=[1, 1, 1, 1, 1],
             ends=[3, 5, 7, 9, 11]),
    ]

    test_data_with_steps = [
        dict(shape=[10, 12], axes=[0, 1], starts=[1, 1], ends=[9, 11], steps=[2, 2]),
        dict(shape=[10, 12], axes=[0, 1], starts=[9, 11], ends=[1, 1], steps=[-1, -1]),
        dict(shape=[10, 12], axes=[0], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[10, 12], axes=[1], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[10, 12], axes=[0, 1], starts=[9, 11], ends=[1, 1], steps=[-2, -2]),
        dict(shape=[8, 10, 12], axes=[0, 1, 2], starts=[1, 1, 1], ends=[7, 9, 11], steps=[2, 2, 2]),
        dict(shape=[8, 10, 12], axes=[0, 1, 2], starts=[7, 9, 11], ends=[1, 1, 1],
             steps=[-1, -1, -1]),
        dict(shape=[8, 10, 12], axes=[0], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[8, 10, 12], axes=[1], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[8, 10, 12], axes=[2], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[8, 10, 12], axes=[0, 1, 2], starts=[7, 9, 11], ends=[1, 1, 1],
             steps=[-2, -2, -2]),
        dict(shape=[6, 8, 10, 12], axes=[0, 1, 2, 3], starts=[1, 1, 1, 1], ends=[5, 7, 9, 11],
             steps=[2, 2, 2, 2]),
        dict(shape=[6, 8, 10, 12], axes=[0, 1, 2, 3], starts=[5, 7, 9, 11], ends=[1, 1, 1, 1],
             steps=[-1, -1, -1, -1]),
        dict(shape=[6, 8, 10, 12], axes=[0], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[6, 8, 10, 12], axes=[1], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[6, 8, 10, 12], axes=[2], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[6, 8, 10, 12], axes=[3], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[6, 8, 10, 12], axes=[0, 1, 2, 3], starts=[5, 7, 9, 11], ends=[1, 1, 1, 1],
             steps=[-2, -2, -2, -2]),
        dict(shape=[4, 6, 8, 10, 12], axes=[0, 1, 2, 3, 4], starts=[1, 1, 1, 1, 1],
             ends=[3, 5, 7, 9, 11],
             steps=[2, 2, 2, 2, 2]),
        dict(shape=[4, 6, 8, 10, 12], axes=[0, 1, 2, 3, 4], starts=[3, 5, 7, 9, 11],
             ends=[1, 1, 1, 1, 1],
             steps=[-1, -1, -1, -1, -1]),
        dict(shape=[4, 6, 8, 10, 12], axes=[0], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[4, 6, 8, 10, 12], axes=[1], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[4, 6, 8, 10, 12], axes=[2], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[4, 6, 8, 10, 12], axes=[3], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[4, 6, 8, 10, 12], axes=[4], starts=[-1], ends=[-9999], steps=[-1]),
        dict(shape=[4, 6, 8, 10, 12], axes=[0, 1, 2, 3, 4], starts=[3, 5, 7, 9, 11],
             ends=[1, 1, 1, 1, 1],
             steps=[-2, -2, -2, -2, -2]),
    ]

    @pytest.mark.parametrize("params", test_data_no_steps)
    @pytest.mark.nightly
    def test_slice_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, opset=6, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_no_steps)
    @pytest.mark.nightly
    def test_slice_const_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, opset=6, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_no_steps + test_data_with_steps)
    @pytest.mark.nightly
    def test_slice_opset10(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip('GREEN_SUITE')
        self._test(
            *self.create_net(**params, opset=10, ir_version=ir_version), ie_device, precision,
            ir_version,
            temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_no_steps + test_data_with_steps)
    @pytest.mark.nightly
    def test_slice_const_opset10(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip('GREEN_SUITE')
        self._test(*self.create_net_const(**params, opset=10, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_no_steps + test_data_with_steps)
    @pytest.mark.nightly
    def test_slice_opset11(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip('GREEN_SUITE')
        self._test(
            *self.create_net(**params, opset=11, ir_version=ir_version), ie_device, precision,
            ir_version,
            temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_no_steps + test_data_with_steps)
    @pytest.mark.nightly
    def test_slice_const_opset11(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, opset=11, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
