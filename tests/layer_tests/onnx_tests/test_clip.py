# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestClip(OnnxRuntimeLayerTest):
    def create_net(self, shape, ir_version, opset, min=None, max=None):
        """
            ONNX net                    IR net

            Input->Clip->Output   =>    Input->Clamp

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

        nodes = []
        if opset < 11:
            args = dict()
            if min is not None:
                args['min'] = min
            if max is not None:
                args['max'] = max
            node_def = onnx.helper.make_node(
                'Clip',
                inputs=['input'],
                outputs=['output'],
                **args
            )
            nodes.append(node_def)
        else:
            clip_inputs = ['input']
            if min is not None:
                node_min_def = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['min_const'],
                    value=helper.make_tensor(
                        name='const_tensor',
                        data_type=TensorProto.FLOAT,
                        dims=[],
                        vals=[min],
                    ),
                )
                clip_inputs.append('min_const')
                nodes.append(node_min_def)
            else:
                clip_inputs.append('')
            if max is not None:
                node_max_def = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['max_const'],
                    value=helper.make_tensor(
                        name='const_tensor',
                        data_type=TensorProto.FLOAT,
                        dims=[],
                        vals=[max],
                    ),
                )
                clip_inputs.append('max_const')
                nodes.append(node_max_def)
            node_def = onnx.helper.make_node(
                'Clip',
                inputs=clip_inputs,
                outputs=['output']
            )
            nodes.append(node_def)

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
            if opset < 11 or min is not None and max is not None:
                nodes_attributes = {
                    'input': {'kind': 'op', 'type': 'Parameter'},
                    'input_data': {'shape': shape, 'kind': 'data'},
                    'node': {'kind': 'op', 'type': 'Clamp',
                             'min': min if min is not None else -3.4028235e+38,
                             'max': max if max is not None else 3.4028235e+38},
                    'node_data': {'shape': shape, 'kind': 'data'},
                    'result': {'kind': 'op', 'type': 'Result'}
                }
                ref_net = build_graph(nodes_attributes,
                                      [('input', 'input_data'),
                                       ('input_data', 'node'),
                                       ('node', 'node_data'),
                                       ('node_data', 'result')
                                       ])
            else:
                nodes_attributes = {
                    'input': {'kind': 'op', 'type': 'Parameter'},
                    'input_data': {'shape': shape, 'kind': 'data'},
                    'input_const_data': {'kind': 'data',
                                         'value': [min] if min is not None else [max]},
                    'const': {'kind': 'op', 'type': 'Const'},
                    'const_data': {'shape': [], 'kind': 'data'},
                    'node': {'kind': 'op', 'type': 'Minimum' if max is not None else 'Maximum'},
                    'node_data': {'shape': shape, 'kind': 'data'},
                    'result': {'kind': 'op', 'type': 'Result'}
                }
                ref_net = build_graph(nodes_attributes,
                                      [('input', 'input_data'),
                                       ('input_const_data', 'const'),
                                       ('const', 'const_data'),
                                       ('input_data', 'node'),
                                       ('const_data', 'node'),
                                       ('node', 'node_data'),
                                       ('node_data', 'result')
                                       ])

        return onnx_net, ref_net

    test_data = [dict(shape=[12], min=-3.5),
                 dict(shape=[12], max=3.5),
                 dict(shape=[12], min=-3.5, max=3.5),
                 dict(shape=[10, 12], min=-3.5),
                 dict(shape=[10, 12], max=3.5),
                 dict(shape=[10, 12], min=-3.5, max=3.5),
                 dict(shape=[8, 10, 12], min=-3.5),
                 dict(shape=[8, 10, 12], max=3.5),
                 dict(shape=[8, 10, 12], min=-3.5, max=3.5),
                 dict(shape=[6, 8, 10, 12], min=-3.5),
                 dict(shape=[6, 8, 10, 12], max=3.5),
                 dict(shape=[6, 8, 10, 12], min=-3.5, max=3.5),
                 dict(shape=[4, 6, 8, 10, 12], min=-3.5),
                 dict(shape=[4, 6, 8, 10, 12], max=3.5),
                 dict(shape=[4, 6, 8, 10, 12], min=-3.5, max=3.5)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_clip_opset6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version, opset=6), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_clip_opset11(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version, opset=11), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)
