# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestPad(OnnxRuntimeLayerTest):
    def create_net(self, shape, mode, pads, value, ir_version, opset=None):
        """
            ONNX net                   IR net

            Input->Pad->Output   =>    Input->Pad

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        _pads = np.array(pads).reshape([2, -1])
        output_shape = (np.array(shape) + _pads[0, :] + _pads[1, :]).tolist()
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        nodes = []
        if opset is not None and opset < 11:
            args = dict(pads=pads)
            if mode:
                args['mode'] = mode
            if value:
                args['value'] = value
            node_def = onnx.helper.make_node(
                'Pad',
                inputs=['input'],
                outputs=['pad'],
                **args
            )
            nodes.append(node_def)
        else:
            node_pads_def = helper.make_node(
                'Constant',
                inputs=[],
                outputs=['pads'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.INT64,
                    dims=[len(pads)],
                    vals=pads,
                ),
            )

            inputs = ['input', 'pads']
            if value is not None:
                node_value_def = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['value'],
                    value=helper.make_tensor(
                        name='const_tensor',
                        data_type=TensorProto.FLOAT,
                        dims=[],
                        vals=[value],
                    ),
                )
                inputs.append('value')
                nodes.append(node_value_def)

            args = dict()
            if mode:
                args['mode'] = mode
            node_def = onnx.helper.make_node(
                'Pad',
                inputs=inputs,
                outputs=['pad'],
                **args
            )
            nodes.extend([node_pads_def, node_def])

        sigmoid_def = onnx.helper.make_node(
            'Elu',
            inputs=['pad'],
            outputs=['output']
        )
        nodes.append(sigmoid_def)

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

        #
        #   Create reference IR net
        #
        ref_net = None
        if check_ir_version(10, None, ir_version):

            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'pads_begin_indata': {'value': _pads[0, :], 'kind': 'data'},
                'pads_begin': {'kind': 'op', 'type': 'Const'},
                'pads_begin_data': {'shape': [len(_pads[0, :])], 'kind': 'data'},
                'pads_end_indata': {'value': _pads[1, :], 'kind': 'data'},
                'pads_end': {'kind': 'op', 'type': 'Const'},
                'pads_end_data': {'shape': [len(_pads[1, :])], 'kind': 'data'},
                'node': {'kind': 'op', 'type': 'Pad', 'pad_mode': 'constant' if not mode else mode},
                'node_data': {'shape': output_shape, 'kind': 'data'},
                'elu': {'kind': 'op', 'type': 'Elu'},
                'elu_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            edges = [('input', 'input_data'),
                     ('input_data', 'node'),
                     ('pads_begin_indata', 'pads_begin'),
                     ('pads_begin', 'pads_begin_data'),
                     ('pads_begin_data', 'node'),
                     ('pads_end_indata', 'pads_end'),
                     ('pads_end', 'pads_end_data'),
                     ('pads_end_data', 'node'),
                     ('node', 'node_data'),
                     ('node_data', 'elu'),
                     ('elu', 'elu_data'),
                     ('elu_data', 'result')
                     ]

            if mode in (None, "constant"):
                nodes_attributes.update({'const_node_indata': {'value': value, 'kind': 'data'},
                                         'const_node': {'kind': 'op', 'type': 'Const'},
                                         'const_node_data': {'shape': None, 'kind': 'data'}
                                         })
                edges += [('const_node_indata', 'const_node'),
                          ('const_node', 'const_node_data'),
                          ('const_node_data', 'node')
                          ]

            ref_net = build_graph(nodes_attributes, edges)

        return onnx_net, ref_net

    test_data_precommit = [
        pytest.param(dict(shape=[6, 8, 10, 12], pads=[1, 2, 3, 4, 5, 6, 7, 8]),
                     marks=pytest.mark.skip(reason="Skipped until fixed")),
        pytest.param(dict(shape=[8, 10, 12, 14, 16], pads=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                     marks=pytest.mark.skip(reason="Skipped until fixed"))]

    test_data = [dict(shape=[10, 12], pads=[1, 1, 1, 1]),
                 dict(shape=[10, 12], pads=[1, 2, 3, 4]),
                 dict(shape=[8, 10, 12], pads=[0, 0, 1, 0, 0, 1]),
                 dict(shape=[8, 10, 12], pads=[1, 2, 3, 4, 5, 6]),
                 dict(shape=[6, 8, 10, 12], pads=[0, 0, 1, 1, 0, 0, 1, 1]),
                 dict(shape=[6, 8, 10, 12], pads=[0, 0, 1, 2, 0, 0, 3, 4]),
                 dict(shape=[6, 8, 10, 12], pads=[1, 1, 1, 1, 1, 1, 1, 1]),
                 dict(shape=[6, 8, 10, 12], pads=[1, 2, 3, 4, 5, 6, 7, 8]),
                 dict(shape=[8, 10, 12, 14, 16], pads=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                 dict(shape=[8, 10, 12, 14, 16], pads=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("mode_value", [(None, None),
                                            (None, 0.5),
                                            ('constant', None),
                                            ('constant', 0.5),
                                            ('reflect', None),
                                            ('edge', None)])
    @pytest.mark.nightly
    def test_pad_opset_9(self, params, mode_value, ie_device, precision, ir_version, temp_dir):
        mode, value = mode_value
        self._test(
            *self.create_net(**params, mode=mode, value=value, ir_version=ir_version, opset=9),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.parametrize("mode_value", [(None, None),
                                            (None, 0.5),
                                            ('constant', None),
                                            ('constant', 0.5),
                                            ('reflect', None),
                                            ('edge', None)])
    @pytest.mark.precommit
    def test_pad_opset_latest_precommit(self, params, mode_value, ie_device, precision, ir_version,
                                        temp_dir):
        mode, value = mode_value
        self._test(*self.create_net(**params, mode=mode, value=value, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("mode_value", [(None, None),
                                            (None, 0.5),
                                            ('constant', None),
                                            ('constant', 0.5),
                                            ('reflect', None),
                                            ('edge', None)])
    @pytest.mark.nightly
    def test_pad_opset_latest(self, params, mode_value, ie_device, precision, ir_version, temp_dir):
        mode, value = mode_value
        self._test(*self.create_net(**params, mode=mode, value=value, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
