# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


def float_array(x):
    return np.array(x, dtype=float)


class TestPooling(OnnxRuntimeLayerTest):
    def create_net(self, shape, kernel_shape, pads, strides, op, ir_version, count_include_pad=None,
                   auto_pad=None,
                   storage_order=None, ceil=False, opset=None):
        """
            ONNX net                      IR net

            Input->Pooling>Output   =>    Input->Pooling

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        node_args = dict(kernel_shape=kernel_shape)
        if auto_pad is not None:
            node_args['auto_pad'] = auto_pad
            if auto_pad == 'VALID':
                pads = np.zeros(len(shape[2:]) * 2, dtype=int)
        else:
            auto_pad = 'NOTSET'
        if count_include_pad is not None:
            node_args['count_include_pad'] = count_include_pad
        else:
            count_include_pad = 0
        if storage_order is not None:
            node_args['storage_order'] = storage_order
        if pads is not None:
            if auto_pad == 'NOTSET':
                node_args['pads'] = pads
            _pads = np.transpose(np.array(pads).reshape([2, -1]))
        else:
            _pads = np.zeros([len(kernel_shape), 2])
        if strides is not None:
            node_args['strides'] = strides
        else:
            strides = np.ones(len(kernel_shape))

        if ceil:
            node_args['ceil_mode'] = 1

        if auto_pad in ['SAME_UPPER', 'SAME_LOWER']:
            out_spacial_shape = np.ceil(np.array(shape[2:], dtype=float) / strides)
        else:
            rounding = np.ceil if ceil else np.floor
            out_spacial_shape = rounding(
                (float_array(shape[2:]) + np.add(_pads[:, 0], _pads[:, 1]) - float_array(
                    kernel_shape)) / strides + 1)

        out_shape = np.array(shape)
        out_shape[2:] = out_spacial_shape
        out_shape = out_shape.astype(int).tolist()
        concat_axis = 0
        out_concat_shape = out_shape.copy()
        out_concat_shape[concat_axis] *= 2
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_concat_shape)

        constant = np.random.randint(-127, 127, out_shape).astype(float)

        node_def = onnx.helper.make_node(
            op,
            inputs=['input'],
            outputs=['pool'],
            **node_args
        )

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

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['pool', 'const1'],
            outputs=['output'],
            axis=concat_axis
        )

        graph_def = helper.make_graph(
            [node_def, node_const_def, node_concat_def],
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
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'node': {'kind': 'op', 'type': None,
                         'pads_begin': _pads[:, 0] if len(shape) > 3 else _pads[0, 0],
                         'pads_end': _pads[:, 1] if len(shape) > 3 else _pads[0, 1],
                         'kernel': kernel_shape[0] if len(kernel_shape) == 1 else kernel_shape,
                         'rounding_type': 'ceil' if auto_pad != 'NOTSET' or ceil else 'floor',
                         'auto_pad': None},
                'node_data': {'shape': out_shape, 'kind': 'data'},
                'node_indicies_data': {'shape': out_shape, 'kind': 'data'},
                'input_const_data': {'kind': 'data', 'value': constant.flatten()},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': out_shape, 'kind': 'data'},
                'concat': {'kind': 'op', 'type': 'Concat', 'axis': concat_axis},
                'concat_data': {'shape': out_concat_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            if op == 'AveragePool':
                nodes_attributes['node']['type'] = 'AvgPool'
                nodes_attributes['node']['exclude-pad'] = True if count_include_pad == 0 else False
            else:
                nodes_attributes['node']['type'] = 'MaxPool'

            edges = [('input', 'input_data'),
                     ('input_data', 'node'),
                     ('node', 'node_data', {'out': 0}),
                     ('input_const_data', 'const'),
                     ('const', 'const_data'),
                     ('node_data', 'concat'),
                     ('const_data', 'concat'),
                     ('concat', 'concat_data'),
                     ('concat_data', 'result')]
            if op == "MaxPool":
                edges.append(('node', 'node_indicies_data', {'out': 1}))
            ref_net = build_graph(nodes_attributes,
                                  edges,
                                  nodes_with_edges_only=True)

        return onnx_net, ref_net

    def create_global_net(self, shape, op, ir_version):
        """
            ONNX net                            IR net

            Input->GlobalPooling>Output   =>    Input->Pooling

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        out_shape = np.ones(len(shape))
        out_shape[:2] = np.array(shape)[:2]
        out_shape = out_shape.astype(int).tolist()
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)

        node_def = onnx.helper.make_node(
            op,
            inputs=['input'],
            outputs=['output']
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

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'input_axes_data': {'kind': 'data', 'value': list(range(2, len(shape)))},
                'axes': {'kind': 'op', 'type': 'Const'},
                'axes_data': {'shape': [len(shape) - 2], 'kind': 'data'},
                'node': {'kind': 'op', 'type': None},
                'node_data': {'shape': out_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            if op == 'GlobalAveragePool':
                nodes_attributes['node']['type'] = 'ReduceMean'
            else:
                nodes_attributes['node']['type'] = 'ReduceMax'

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node'),
                                   ('input_axes_data', 'axes'),
                                   ('axes', 'axes_data'),
                                   ('axes_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')
                                   ])

        return onnx_net, ref_net

    test_data_precommit = [
        dict(shape=[2, 3, 10], kernel_shape=[2], pads=None, strides=[3]),
        dict(shape=[2, 3, 30, 30], kernel_shape=[5, 5], pads=None, strides=[3, 2]),
        dict(shape=[2, 3, 28, 28, 28], kernel_shape=[5, 5, 5], pads=[2, 4, 2, 0, 0, 2],
             strides=None),
        dict(shape=[2, 3, 30, 30, 30], kernel_shape=[5, 5, 5], pads=None, strides=[3, 3, 5])]

    test_data = [
        dict(shape=[2, 3, 10], kernel_shape=[2], pads=None, strides=None),
        dict(shape=[2, 3, 10], kernel_shape=[2], pads=[2, 2], strides=None),
        dict(shape=[2, 3, 10], kernel_shape=[2], pads=None, strides=[3]),
        dict(shape=[2, 3, 30, 30], kernel_shape=[2, 2], pads=None, strides=None),
        dict(shape=[2, 3, 30, 30], kernel_shape=[4, 2], pads=None, strides=None),
        dict(shape=[2, 3, 30, 30], kernel_shape=[2, 4], pads=None, strides=None),
        dict(shape=[2, 3, 28, 28], kernel_shape=[3, 3], pads=[2, 2, 2, 2], strides=None),
        dict(shape=[2, 3, 28, 28], kernel_shape=[5, 5], pads=[0, 2, 0, 4], strides=None),
        dict(shape=[2, 3, 28, 28], kernel_shape=[5, 5], pads=[2, 0, 4, 0], strides=None),
        dict(shape=[2, 3, 30, 30], kernel_shape=[5, 5], pads=None, strides=[3, 3]),
        dict(shape=[2, 3, 30, 30], kernel_shape=[5, 5], pads=None, strides=[2, 3]),
        dict(shape=[2, 3, 30, 30], kernel_shape=[5, 5], pads=None, strides=[3, 2]),
        dict(shape=[2, 3, 30, 30, 30], kernel_shape=[2, 2, 2], pads=None, strides=None),
        dict(shape=[2, 3, 30, 30, 30], kernel_shape=[4, 2, 2], pads=None, strides=None),
        dict(shape=[2, 3, 30, 30, 30], kernel_shape=[2, 4, 2], pads=None, strides=None),
        dict(shape=[2, 3, 30, 30, 30], kernel_shape=[2, 2, 4], pads=None, strides=None),
        dict(shape=[2, 3, 28, 28, 28], kernel_shape=[3, 3, 3], pads=[2, 2, 2, 2, 2, 2],
             strides=None),
        dict(shape=[2, 3, 28, 28, 28], kernel_shape=[5, 5, 5], pads=[2, 4, 2, 0, 0, 2],
             strides=None),
        dict(shape=[2, 3, 30, 30, 30], kernel_shape=[5, 5, 5], pads=None, strides=[3, 3, 3]),
        dict(shape=[2, 3, 30, 30, 30], kernel_shape=[5, 5, 5], pads=None, strides=[5, 3, 3]),
        dict(shape=[2, 3, 30, 30, 30], kernel_shape=[5, 5, 5], pads=None, strides=[3, 5, 3]),
        dict(shape=[2, 3, 30, 30, 30], kernel_shape=[5, 5, 5], pads=None, strides=[3, 3, 5])]

    test_data_autopad_precommit = [
        dict(shape=[2, 3, 30, 30, 30], auto_pad='VALID', kernel_shape=[2, 2, 4], pads=None,
             strides=None),
        dict(shape=[2, 3, 21, 21, 21], auto_pad='VALID', kernel_shape=[3, 3, 3], pads=None,
             strides=[3, 2, 3]),
        dict(shape=[2, 3, 21, 21, 21], auto_pad='VALID', kernel_shape=[3, 3, 3], pads=None,
             strides=[3, 3, 2])]

    test_data_autopad = [
        dict(shape=[2, 3, 10], auto_pad='SAME_UPPER', kernel_shape=[2], pads=[0, 1], strides=[3]),
        dict(shape=[2, 3, 10], auto_pad='SAME_LOWER', kernel_shape=[2], pads=[0, 1], strides=[3]),
        dict(shape=[2, 3, 10], auto_pad='VALID', kernel_shape=[2], pads=None, strides=[3]),
        dict(shape=[2, 3, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[2, 2], pads=[0, 0, 1, 1],
             strides=None),
        dict(shape=[2, 3, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[4, 2], pads=[1, 0, 2, 1],
             strides=None),
        dict(shape=[2, 3, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[2, 4], pads=[0, 1, 1, 2],
             strides=None),
        dict(shape=[2, 3, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[5, 5], pads=[1, 1, 1, 1],
             strides=[3, 3]),
        dict(shape=[2, 3, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[5, 5], pads=[1, 1, 2, 1],
             strides=[2, 3]),
        dict(shape=[2, 3, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[5, 5], pads=[1, 1, 1, 2],
             strides=[3, 2]),
        dict(shape=[2, 3, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[2, 2], pads=[0, 0, 1, 1],
             strides=None),
        dict(shape=[2, 3, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[4, 2], pads=[1, 0, 2, 1],
             strides=None),
        dict(shape=[2, 3, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[2, 4], pads=[0, 1, 1, 2],
             strides=None),
        dict(shape=[2, 3, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[5, 5], pads=[1, 1, 1, 1],
             strides=[3, 3]),
        dict(shape=[2, 3, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[5, 5], pads=[1, 1, 2, 1],
             strides=[2, 3]),
        dict(shape=[2, 3, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[5, 5], pads=[1, 1, 1, 2],
             strides=[3, 2]),
        dict(shape=[2, 3, 30, 30], auto_pad='VALID', kernel_shape=[2, 2], pads=None, strides=None),
        dict(shape=[2, 3, 30, 30], auto_pad='VALID', kernel_shape=[4, 2], pads=None, strides=None),
        dict(shape=[2, 3, 30, 30], auto_pad='VALID', kernel_shape=[2, 4], pads=None, strides=None),
        dict(shape=[2, 3, 21, 21], auto_pad='VALID', kernel_shape=[3, 3], pads=None,
             strides=[3, 3]),
        dict(shape=[2, 3, 21, 21], auto_pad='VALID', kernel_shape=[3, 3], pads=None,
             strides=[2, 3]),
        dict(shape=[2, 3, 21, 21], auto_pad='VALID', kernel_shape=[3, 3], pads=None,
             strides=[3, 2]),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[2, 2, 2],
             pads=[0, 0, 0, 1, 1, 1],
             strides=None),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[4, 2, 2],
             pads=[1, 0, 0, 2, 1, 1],
             strides=None),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[2, 4, 2],
             pads=[0, 1, 0, 1, 2, 1],
             strides=None),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[2, 2, 4],
             pads=[0, 0, 1, 1, 1, 2],
             strides=None),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[5, 5, 5],
             pads=[1, 1, 1, 1, 1, 1],
             strides=[3, 3, 3]),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[5, 5, 5],
             pads=[0, 1, 1, 0, 1, 1],
             strides=[5, 3, 3]),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[5, 5, 5],
             pads=[1, 0, 1, 1, 0, 1],
             strides=[3, 5, 3]),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_UPPER', kernel_shape=[5, 5, 5],
             pads=[1, 1, 0, 1, 1, 0],
             strides=[3, 3, 5]),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[2, 2, 2],
             pads=[0, 0, 0, 1, 1, 1],
             strides=None),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[4, 2, 2],
             pads=[1, 0, 0, 2, 1, 1],
             strides=None),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[2, 4, 2],
             pads=[0, 1, 0, 1, 2, 1],
             strides=None),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[2, 2, 4],
             pads=[0, 0, 1, 1, 1, 2],
             strides=None),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[5, 5, 5],
             pads=[1, 1, 1, 1, 1, 1],
             strides=[3, 3, 3]),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[5, 5, 5],
             pads=[0, 1, 1, 0, 1, 1],
             strides=[5, 3, 3]),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[5, 5, 5],
             pads=[1, 0, 1, 1, 0, 1],
             strides=[3, 5, 3]),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='SAME_LOWER', kernel_shape=[5, 5, 5],
             pads=[1, 1, 0, 1, 1, 0],
             strides=[3, 3, 5]),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='VALID', kernel_shape=[2, 2, 2], pads=None,
             strides=None),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='VALID', kernel_shape=[4, 2, 2], pads=None,
             strides=None),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='VALID', kernel_shape=[2, 4, 2], pads=None,
             strides=None),
        dict(shape=[2, 3, 30, 30, 30], auto_pad='VALID', kernel_shape=[2, 2, 4], pads=None,
             strides=None),
        dict(shape=[2, 3, 21, 21, 21], auto_pad='VALID', kernel_shape=[3, 3, 3], pads=None,
             strides=[3, 3, 3]),
        dict(shape=[2, 3, 21, 21, 21], auto_pad='VALID', kernel_shape=[3, 3, 3], pads=None,
             strides=[2, 3, 3]),
        dict(shape=[2, 3, 21, 21, 21], auto_pad='VALID', kernel_shape=[3, 3, 3], pads=None,
             strides=[3, 2, 3]),
        dict(shape=[2, 3, 21, 21, 21], auto_pad='VALID', kernel_shape=[3, 3, 3], pads=None,
             strides=[3, 3, 2])]

    global_test_data = [dict(shape=[2, 3, 10]),
                        dict(shape=[2, 3, 32, 32]),
                        dict(shape=[2, 3, 32, 32, 32])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("incl_pad", [None, 1])
    @pytest.mark.nightly
    def test_avgpool_opset7(self, params, incl_pad, ie_device, precision, ir_version, temp_dir):
        if not len(params['shape']) in [4, 5]:
            pytest.skip("Pooling layer support only 4D and 5D input tensors")
        self._test(
            *self.create_net(**params, op='AveragePool', count_include_pad=incl_pad,
                             ir_version=ir_version, opset=7),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_autopad)
    @pytest.mark.nightly
    def test_avgpool_opset7_autopad(self, params, ie_device, precision, ir_version, temp_dir):
        if not len(params['shape']) in [4, 5]:
            pytest.skip("Pooling layer support only 4D and 5D input tensors")
        self._test(*self.create_net(**params, op='AveragePool', ir_version=ir_version, opset=7),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("incl_pad", [None, 1])
    @pytest.mark.parametrize("ceil", [True, False])
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_avgpool_opset10(self, params, incl_pad, ceil, ie_device, precision, ir_version,
                             temp_dir):
        if not len(params['shape']) in [4, 5]:
            pytest.skip("Pooling layer support only 4D and 5D input tensors")
        self._test(
            *self.create_net(**params, op='AveragePool', count_include_pad=incl_pad, ceil=ceil,
                             ir_version=ir_version,
                             opset=10), ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_autopad)
    @pytest.mark.nightly
    def test_avgpool_opset10_autopad(self, params, ie_device, precision, ir_version, temp_dir):
        if not len(params['shape']) in [4, 5]:
            pytest.skip("Pooling layer support only 4D and 5D input tensors")
        self._test(*self.create_net(**params, op='AveragePool', ir_version=ir_version, opset=10),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("st_order", [None, 1])
    @pytest.mark.nightly
    def test_maxpool_opset8(self, params, st_order, ie_device, precision, ir_version, temp_dir):
        if not len(params['shape']) in [4, 5]:
            pytest.skip("Pooling layer support only 4D and 5D input tensors")
        self._test(
            *self.create_net(**params, op='MaxPool', storage_order=st_order, ir_version=ir_version,
                             opset=8),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_autopad)
    @pytest.mark.nightly
    def test_maxpool_opset8_autopad(self, params, ie_device, precision, ir_version, temp_dir):
        if not len(params['shape']) in [4, 5]:
            pytest.skip("Pooling layer support only 4D and 5D input tensors")
        self._test(*self.create_net(**params, op='MaxPool', ir_version=ir_version, opset=8),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("st_order", [None, 1])
    @pytest.mark.parametrize("ceil", [True, False])
    @pytest.mark.nightly
    def test_maxpool_opset10(self, params, st_order, ceil, ie_device, precision, ir_version,
                             temp_dir):
        if not len(params['shape']) in [4, 5]:
            pytest.skip("Pooling layer support only 4D and 5D input tensors")
        self._test(*self.create_net(**params, op='MaxPool', storage_order=st_order, ceil=ceil,
                                    ir_version=ir_version,
                                    opset=10), ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_autopad_precommit)
    @pytest.mark.precommit
    def test_maxpool_opset10_autopad(self, params, ie_device, precision, ir_version, temp_dir):
        if not len(params['shape']) in [4, 5]:
            pytest.skip("Pooling layer support only 4D and 5D input tensors")
        self._test(*self.create_net(**params, op='MaxPool', ir_version=ir_version, opset=10),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_autopad)
    @pytest.mark.nightly
    def test_maxpool_opset10_autopad(self, params, ie_device, precision, ir_version, temp_dir):
        if not len(params['shape']) in [4, 5]:
            pytest.skip("Pooling layer support only 4D and 5D input tensors")
        self._test(*self.create_net(**params, op='MaxPool', ir_version=ir_version, opset=10),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", global_test_data)
    @pytest.mark.nightly
    def test_global_avgpool(self, params, ie_device, precision, ir_version, temp_dir):
        if not len(params['shape']) in [4, 5]:
            pytest.skip("Pooling layer support only 4D and 5D input tensors")
        self._test(*self.create_global_net(**params, op='GlobalAveragePool', ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", global_test_data)
    @pytest.mark.nightly
    def test_global_maxpool(self, params, ie_device, precision, ir_version, temp_dir):
        if not len(params['shape']) in [4, 5]:
            pytest.skip("Pooling layer support only 4D and 5D input tensors")
        self._test(*self.create_global_net(**params, op='GlobalMaxPool', ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
