# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest


class TestLoop(OnnxRuntimeLayerTest):

    def create_loop(self):
        """
            ONNX net

            Input->Loop->Output   =>   Only accuracy check

        """

        #   Create ONNX model

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input_shape = [1, 4, 64, 54]

        in_1 = helper.make_tensor_value_info('IN_1', TensorProto.FLOAT, input_shape)
        in_1_int = helper.make_tensor_value_info('in_1_int', TensorProto.FLOAT, input_shape)
        in_1_int_out = helper.make_tensor_value_info('in_1_int_out', TensorProto.FLOAT, input_shape)

        out_1 = helper.make_tensor_value_info('OUT_1', TensorProto.FLOAT, None)
        res = helper.make_tensor_value_info('res', TensorProto.FLOAT, None)

        m_1 = helper.make_tensor_value_info('m_1', TensorProto.INT64, [1])

        cond_int_1 = helper.make_tensor_value_info('cond_int_1', TensorProto.BOOL, [1])
        cond_out_1 = helper.make_tensor_value_info('cond_out_1', TensorProto.BOOL, [1])

        m_1_value = np.array([10], dtype=np.int64)
        cond_value = np.array([True], np.bool)
        one_value = np.ones(input_shape, dtype=np.float)

        M_1 = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['M_1'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=m_1_value.shape,
                vals=m_1_value.flatten().astype(np.int64),
            ),
        )

        cond = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['cond'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.BOOL,
                dims=cond_value.shape,
                vals=cond_value.flatten().astype(np.bool),
            ),
        )

        one = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['one'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=one_value.shape,
                vals=one_value.flatten().astype(np.float),
            ),
        )

        one_int = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['one_int'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=one_value.shape,
                vals=one_value.flatten().astype(np.int64),
            ),
        )

        add_one_node = onnx.helper.make_node(
            'Add',
            inputs=['in_1_int', 'one'],
            outputs=['OUT_1']
        )

        add_one_to_m_node = onnx.helper.make_node(
            'Add',
            inputs=['m_1', 'one_int'],
            outputs=['m_1_loop_1']
        )

        cond_1_node = onnx.helper.make_node(
            'Identity',
            inputs=['cond_int_1'],
            outputs=['cond_out_1'],
        )

        in_1_int_node = onnx.helper.make_node(
            'Identity',
            inputs=['in_1_int'],
            outputs=['in_1_int_out'],
        )

        body_graph_1 = helper.make_graph(
            [one, one_int, add_one_node, add_one_to_m_node, cond_1_node, in_1_int_node],
            'body_graph_1',
            [m_1, cond_int_1, in_1_int],
            [cond_out_1, in_1_int_out, out_1],
        )

        node_loop_1 = onnx.helper.make_node(
            'Loop',
            inputs=['M_1', 'cond', 'IN_1'],
            outputs=['cond_out_1', 'OUT_1'],
            body=body_graph_1
        )

        res_node = onnx.helper.make_node(
            'Identity',
            inputs=['OUT_1'],
            outputs=['res'],
        )

        graph_def = helper.make_graph(
            [M_1, cond, node_loop_1, res_node],
            'graph',
            [in_1],
            [res],
        )

        onnx_net = helper.make_model(graph_def, producer_name='test_loop_model')
        # We do not create reference graph, as it's too complicated to construct it
        # So we return None to skip IR comparision
        return onnx_net, None

    def create_loop_in_loop(self):
        """
            ONNX net

            Input->Loop(Loop)->Output   =>   Only accuracy check

        """

        #   Create ONNX model

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input_shape = [1, 4, 64, 54]

        in_1 = helper.make_tensor_value_info('IN_1', TensorProto.FLOAT, input_shape)
        in_1_int = helper.make_tensor_value_info('in_1_int', TensorProto.FLOAT, input_shape)
        in_1_int_out = helper.make_tensor_value_info('in_1_int_out', TensorProto.FLOAT, input_shape)

        in_2 = helper.make_tensor_value_info('IN_2', TensorProto.FLOAT, input_shape)
        in_2_int = helper.make_tensor_value_info('in_2_int', TensorProto.FLOAT, input_shape)
        in_2_int_out = helper.make_tensor_value_info('in_2_int_out', TensorProto.FLOAT, input_shape)

        out_1 = helper.make_tensor_value_info('OUT_1', TensorProto.FLOAT, None)
        out_2 = helper.make_tensor_value_info('OUT_2', TensorProto.FLOAT, None)
        res = helper.make_tensor_value_info('res', TensorProto.FLOAT, None)

        m_1 = helper.make_tensor_value_info('m_1', TensorProto.INT64, [1])
        m_2 = helper.make_tensor_value_info('m_2', TensorProto.INT64, [1])

        cond_int_1 = helper.make_tensor_value_info('cond_int_1', TensorProto.BOOL, [1])
        cond_out_1 = helper.make_tensor_value_info('cond_out_1', TensorProto.BOOL, [1])
        cond_int_2 = helper.make_tensor_value_info('cond_int_2', TensorProto.BOOL, [1])
        cond_out_2 = helper.make_tensor_value_info('cond_out_2', TensorProto.BOOL, [1])

        m_1_value = np.array([10], dtype=np.int64)
        m_2_value = np.array([5], dtype=np.int64)
        cond_value = np.array([True], np.bool)
        one_value = np.ones(input_shape, dtype=np.float)

        M_1 = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['M_1'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=m_1_value.shape,
                vals=m_1_value.flatten().astype(np.int64),
            ),
        )

        M_2 = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['M_2'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=m_2_value.shape,
                vals=m_2_value.flatten().astype(np.int64),
            ),
        )

        cond = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['cond'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.BOOL,
                dims=cond_value.shape,
                vals=cond_value.flatten().astype(np.bool),
            ),
        )

        one = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['one'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=one_value.shape,
                vals=one_value.flatten().astype(np.float),
            ),
        )

        one_int = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['one_int'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=one_value.shape,
                vals=one_value.flatten().astype(np.int64),
            ),
        )

        add_one_node = onnx.helper.make_node(
            'Add',
            inputs=['in_1_int', 'one'],
            outputs=['in_1_loop_1']
        )

        add_one_to_m_node = onnx.helper.make_node(
            'Add',
            inputs=['m_1', 'one_int'],
            outputs=['m_1_loop_1']
        )

        cond_2 = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['cond_2'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.BOOL,
                dims=cond_value.shape,
                vals=cond_value.flatten().astype(np.bool),
            ),
        )

        one_int_in_2 = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['one_int_in_2'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=one_value.shape,
                vals=one_value.flatten().astype(np.int64),
            ),
        )

        one_in_2 = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['one_in_2'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=one_value.shape,
                vals=one_value.flatten().astype(np.float),
            ),
        )

        add_one_to_m_2_node = onnx.helper.make_node(
            'Add',
            inputs=['m_2', 'one_int_in_2'],
            outputs=['m_2_loop_2']
        )

        add_node = onnx.helper.make_node(
            'Add',
            inputs=['in_2_int', 'one_in_2'],
            outputs=['OUT_2']
        )

        cond_2_node = onnx.helper.make_node(
            'Identity',
            inputs=['cond_int_2'],
            outputs=['cond_out_2'],
        )

        in_2_int_node = onnx.helper.make_node(
            'Identity',
            inputs=['in_2_int'],
            outputs=['in_2_int_out'],
        )

        body_graph_2 = helper.make_graph(
            [one_in_2, add_node, cond_2_node, in_2_int_node, one_int_in_2, add_one_to_m_2_node],
            'body_graph_2',
            [m_2, cond_int_2, in_2_int],
            [cond_out_2, in_2_int_out, out_2],
        )

        node_loop_2 = onnx.helper.make_node(
            'Loop',
            inputs=['M_2', 'cond_2', 'IN_2'],
            outputs=['cond_out_2', 'OUT_2'],
            body=body_graph_2
        )

        out_1_node = onnx.helper.make_node(
            'Identity',
            inputs=['OUT_2'],
            outputs=['OUT_1'],
        )

        cond_1_node = onnx.helper.make_node(
            'Identity',
            inputs=['cond_int_1'],
            outputs=['cond_out_1'],
        )

        in_1_int_node = onnx.helper.make_node(
            'Identity',
            inputs=['in_1_int'],
            outputs=['in_1_int_out'],
        )

        body_graph_1 = helper.make_graph(
            [one, add_one_node, one_int, add_one_to_m_node, M_2, cond_2, node_loop_2, out_1_node, cond_1_node,
             in_1_int_node],
            'body_graph_1',
            [m_1, cond_int_1, in_1_int],
            [cond_out_1, in_1_int_out, out_1],
        )

        node_loop_1 = onnx.helper.make_node(
            'Loop',
            inputs=['M_1', 'cond', 'IN_1'],
            outputs=['cond_out_1', 'OUT_1'],
            body=body_graph_1
        )

        res_node = onnx.helper.make_node(
            'Identity',
            inputs=['OUT_1'],
            outputs=['res'],
        )

        graph_def = helper.make_graph(
            [M_1, cond, node_loop_1, res_node],
            'graph',
            [in_1, in_2],
            [res],
        )

        onnx_net = helper.make_model(graph_def, producer_name='test_loop_in_loop_model')
        # We do not create reference graph, as it's too complicated to construct it
        # So we return None to skip IR comparision

        return onnx_net, None

    @pytest.mark.precommit
    @pytest.mark.timeout(250)
    def test_loop_simple_precommit(self, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_loop(), ie_device, precision, ir_version, temp_dir=temp_dir,
                   infer_timeout=150)

    @pytest.mark.precommit
    @pytest.mark.timeout(250)
    def test_loop_in_loop_simple_precommit(self, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_loop_in_loop(), ie_device, precision, ir_version, temp_dir=temp_dir,
                   infer_timeout=150)
