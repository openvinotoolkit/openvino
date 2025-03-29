# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestReduceL1L2(OnnxRuntimeLayerTest):
    def create_reduce_lp(self, shape, axes, keep_dims, reduce_p, ir_version):
        """
            ONNX net                              IR net

            Input->ReduceLX(axes)->Output   =>    Input->ReduceLX

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto, OperatorSetIdProto

        output_shape = shape.copy()
        _axes = axes.copy() if axes is not None else list(range(len(shape)))
        for axis in _axes:
            output_shape[axis] = 1

        if not keep_dims:
            output_shape = [dim for dim in output_shape if dim != 1]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        args = dict(keepdims=keep_dims)
        if axes:
            args['axes'] = axes
        node_def = onnx.helper.make_node(
            "ReduceL" + str(reduce_p),
            inputs=['input'],
            outputs=['output'],
            **args
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            'test_model',
            [input],
            [output],
        )

        # Set ONNX Opset
        onnx_opset = OperatorSetIdProto()
        onnx_opset.domain = ""
        # ONNX opset with `axes` as attribute in ONNX Reduce ops
        onnx_opset.version = 11

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model', opset_imports=[onnx_opset])

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'input_data_1': {'shape': [len(_axes)], 'value': _axes, 'kind': 'data'},
                'const_1': {'kind': 'op', 'type': 'Const'},
                'const_data_1': {'shape': [len(_axes)], 'kind': 'data'},
                'reduce': {'kind': 'op', 'type': "ReduceL" + str(reduce_p), 'keep_dims': keep_dims},
                'reduce_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data_1', 'const_1'),
                                   ('const_1', 'const_data_1'),
                                   ('input_data', 'reduce'),
                                   ('const_data_1', 'reduce'),
                                   ('reduce', 'reduce_data'),
                                   ('reduce_data', 'result')
                                   ])

        return onnx_net, ref_net

    def create_reduce_lp_const(self, shape, axes, keep_dims, reduce_p, ir_version):
        """
            ONNX net                              IR net

            Input->ReduceLX(axes)->Output   =>    Input->ReduceLX

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto, OperatorSetIdProto

        output_shape = shape.copy()
        _axes = axes.copy() if axes is not None else list(range(len(shape)))
        for axis in _axes:
            output_shape[axis] = 1

        if not keep_dims:
            output_shape = [dim for dim in output_shape if dim != 1]
        if len(output_shape) == 0:
            output_shape = [1]

        concat_axis = 0
        concat_output_shape = output_shape.copy()
        concat_output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, output_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, concat_output_shape)

        constant = np.random.randn(*shape).astype(float)

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

        args = dict(keepdims=keep_dims)
        if axes:
            args['axes'] = axes
        node_def = onnx.helper.make_node(
            "ReduceL" + str(reduce_p),
            inputs=['const1'],
            outputs=['reduce'],
            **args
        )

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['input', 'reduce'],
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

        # Set ONNX Opset
        onnx_opset = OperatorSetIdProto()
        onnx_opset.domain = ""
        # ONNX opset with `axes` as attribute in ONNX Reduce ops
        onnx_opset.version = 11

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model', opset_imports=[onnx_opset])

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #
        constant = np.power(
            np.sum(a=np.abs(np.power(constant, reduce_p)), axis=tuple(_axes), keepdims=keep_dims),
            1 / reduce_p)
        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': output_shape, 'kind': 'data'},
                'input_const_data': {'kind': 'data', 'value': constant.flatten()},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': constant.shape, 'kind': 'data'},
                'concat': {'kind': 'op', 'type': 'Concat', 'axis': concat_axis},
                'concat_data': {'shape': concat_output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_const_data', 'const'),
                                   ('const', 'const_data'),
                                   ('input_data', 'concat'),
                                   ('const_data', 'concat'),
                                   ('concat', 'concat_data'),
                                   ('concat_data', 'result')
                                   ])

        return onnx_net, ref_net

    test_data_precommit = [
        dict(shape=[2, 4, 6, 8], axes=[-3, -1, -2]),
        dict(shape=[2, 4, 6, 8, 10], axes=[-4, -2]),
    ]

    test_data = [
        dict(shape=[8], axes=None),
        dict(shape=[8], axes=[0]),
        dict(shape=[2, 4, 6], axes=None),
        dict(shape=[2, 4, 6], axes=[1]),
        dict(shape=[2, 4, 6], axes=[-2]),
        dict(shape=[2, 4, 6], axes=[-2, -1]),
        dict(shape=[2, 4, 6, 8], axes=[0]),
        dict(shape=[2, 4, 6, 8], axes=[-3, -1, -2]),
        dict(shape=[2, 4, 6, 8, 10], axes=None),
        dict(shape=[2, 4, 6, 8, 10], axes=[-2]),
        dict(shape=[2, 4, 6, 8, 10], axes=[1, 3]),
        dict(shape=[2, 4, 6, 8, 10], axes=[-4, -2]),
    ]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.parametrize("reduce_p", [1, 2])
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() in ('Darwin', 'Linux') and platform.machine() in ('arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'),
                       reason='Ticket - 122846, 122783, 126312')
    def test_reduce_lp_precommit(self, params, keep_dims, reduce_p, ie_device, precision,
                                 ir_version, temp_dir):
        self._test(*self.create_reduce_lp(**params, keep_dims=keep_dims, reduce_p=reduce_p,
                                          ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.parametrize("reduce_p", [1, 2])
    @pytest.mark.nightly
    def test_reduce_lp(self, params, keep_dims, reduce_p, ie_device, precision, ir_version,
                       temp_dir):
        if ie_device == 'GPU':
            pytest.skip('GREEN_SUITE')
        self._test(*self.create_reduce_lp(**params, keep_dims=keep_dims, reduce_p=reduce_p,
                                          ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.parametrize("reduce_p", [1, 2])
    @pytest.mark.precommit
    def test_reduce_lp_const_precommit(self, params, keep_dims, reduce_p, ie_device, precision,
                                       ir_version, temp_dir):
        self._test(
            *self.create_reduce_lp_const(**params, keep_dims=keep_dims, reduce_p=reduce_p,
                                         ir_version=ir_version),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.parametrize("reduce_p", [1, 2])
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_reduce_lp_const(self, params, keep_dims, reduce_p, ie_device, precision, ir_version,
                             temp_dir):
        self._test(*self.create_reduce_lp_const(**params, keep_dims=keep_dims, reduce_p=reduce_p,
                                                ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
