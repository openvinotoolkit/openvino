# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestReduce(OnnxRuntimeLayerTest):
    def create_reduce(self, shape, reshapped_shape, op, axes, keep_dims, ir_version):
        """
            ONNX net                               IR net

            Input->Reduce Operation (axes)->Output   =>    Input->Reduce Operation

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto, OperatorSetIdProto

        if op not in ['ReduceMin', 'ReduceMax', 'ReduceMean', 'ReduceProd', 'ReduceSum']:
            raise ValueError("Operation has to be either Reduce(Min or Max or Mean or Sum or Prod")

        output_shape = shape.copy()
        for axis in axes:
            output_shape[axis] = 1

        if not keep_dims:
            output_shape = [dim for dim in output_shape if dim != 1]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        node_def = onnx.helper.make_node(
            op,
            inputs=['input'],
            outputs=['output'],
            axes=axes,
            keepdims=keep_dims
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
                'input_data_1': {'shape': [len(axes)], 'value': axes, 'kind': 'data'},
                'const_1': {'kind': 'op', 'type': 'Const'},
                'const_data_1': {'shape': [len(axes)], 'kind': 'data'},
                'reduce': {'kind': 'op', 'type': op, 'keep_dims': keep_dims},
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

    test_data_precommit = [
        dict(shape=[2, 4, 6], reshapped_shape=[2, 1, 4 * 6, 1], axes=[1, 2]),
        dict(shape=[2, 4, 6, 8], reshapped_shape=[2, 1, 4 * 6 * 8, 1], axes=[1, 2, 3]),
        dict(shape=[2, 4, 6, 8, 10], reshapped_shape=[2, 4, 6 * 8 * 10, 1], axes=[2, 3, 4])
    ]

    test_data = [
        dict(shape=[2, 4, 6], reshapped_shape=[1, 1, 2, 4 * 6], axes=[0]),
        dict(shape=[2, 4, 6], reshapped_shape=[2, 1, 4, 6], axes=[1]),
        dict(shape=[2, 4, 6], reshapped_shape=[2, 4, 6, 1], axes=[2]),
        dict(shape=[2, 4, 6], reshapped_shape=[1, 1, 2 * 4, 6], axes=[0, 1]),
        dict(shape=[2, 4, 6], reshapped_shape=[2, 1, 4 * 6, 1], axes=[1, 2]),
        dict(shape=[2, 4, 6, 8], reshapped_shape=[1, 1, 2, 4 * 6 * 8], axes=[0]),
        dict(shape=[2, 4, 6, 8], reshapped_shape=[2, 1, 4, 6 * 8], axes=[1]),
        dict(shape=[2, 4, 6, 8], reshapped_shape=[2, 4, 6, 8], axes=[2]),
        dict(shape=[2, 4, 6, 8], reshapped_shape=[2, 4 * 6, 8, 1], axes=[3]),
        dict(shape=[2, 4, 6, 8], reshapped_shape=[1, 1, 2 * 4, 6 * 8], axes=[0, 1]),
        dict(shape=[2, 4, 6, 8], reshapped_shape=[2, 1, 4 * 6, 8], axes=[1, 2]),
        dict(shape=[2, 4, 6, 8], reshapped_shape=[2, 4, 6 * 8, 1], axes=[2, 3]),
        dict(shape=[2, 4, 6, 8], reshapped_shape=[1, 1, 2 * 4 * 6, 8], axes=[0, 1, 2]),
        dict(shape=[2, 4, 6, 8], reshapped_shape=[2, 1, 4 * 6 * 8, 1], axes=[1, 2, 3]),
        dict(shape=[2, 4, 6, 8, 10], reshapped_shape=[1, 1, 2, 4 * 6 * 8 * 10], axes=[0]),
        dict(shape=[2, 4, 6, 8, 10], reshapped_shape=[2, 1, 4, 6 * 8 * 10], axes=[1]),
        dict(shape=[2, 4, 6, 8, 10], reshapped_shape=[2, 4, 6, 8 * 10], axes=[2]),
        dict(shape=[2, 4, 6, 8, 10], reshapped_shape=[2, 4 * 6, 8, 10], axes=[3]),
        dict(shape=[2, 4, 6, 8, 10], reshapped_shape=[2, 4 * 6 * 8, 10, 1], axes=[4]),
        dict(shape=[2, 4, 6, 8, 10], reshapped_shape=[1, 1, 2 * 4, 6 * 8 * 10], axes=[0, 1]),
        dict(shape=[2, 4, 6, 8, 10], reshapped_shape=[2, 1, 4 * 6, 8 * 10], axes=[1, 2]),
        dict(shape=[2, 4, 6, 8, 10], reshapped_shape=[2, 4, 6 * 8, 10], axes=[2, 3]),
        dict(shape=[2, 4, 6, 8, 10], reshapped_shape=[2, 4 * 6, 8 * 10, 1], axes=[3, 4]),
        dict(shape=[2, 4, 6, 8, 10], reshapped_shape=[1, 1, 2 * 4 * 6, 8 * 10], axes=[0, 1, 2]),
        dict(shape=[2, 4, 6, 8, 10], reshapped_shape=[2, 4, 6 * 8 * 10, 1], axes=[2, 3, 4])
    ]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.precommit
    def test_reduce_max_precommit(self, params, keep_dims, ie_device, precision, ir_version,
                                  temp_dir):
        self._test(*self.create_reduce(**params, op='ReduceMax', keep_dims=keep_dims,
                                       ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.nightly
    def test_reduce_max(self, params, keep_dims, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reduce(**params, op='ReduceMax', keep_dims=keep_dims,
                                       ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_reduce_sum(self, params, keep_dims, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reduce(**params, op='ReduceSum', keep_dims=keep_dims,
                                       ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_reduce_prod(self, params, keep_dims, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reduce(**params, op='ReduceProd', keep_dims=keep_dims,
                                       ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.precommit
    def test_reduce_mean_precommit(self, params, keep_dims, ie_device, precision, ir_version,
                                   temp_dir):
        self._test(*self.create_reduce(**params, op='ReduceMean', keep_dims=keep_dims,
                                       ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_reduce_mean(self, params, keep_dims, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reduce(**params, op='ReduceMean', keep_dims=keep_dims,
                                       ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.precommit
    def test_reduce_min_precommit(self, params, keep_dims, ie_device, precision, ir_version,
                                  temp_dir):
        self._test(*self.create_reduce(**params, op='ReduceMin', keep_dims=keep_dims,
                                       ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.nightly
    def test_reduce_min(self, params, keep_dims, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reduce(**params, op='ReduceMin', keep_dims=keep_dims,
                                       ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
