# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestLoop(OnnxRuntimeLayerTest):
    @staticmethod
    def create_const(name, tensor_type, value):
        from onnx import helper
        from onnx import TensorProto

        if tensor_type == TensorProto.INT64:
            np_type = np.int64
        elif tensor_type == TensorProto.FLOAT:
            np_type = np.float32
        elif tensor_type == TensorProto.BOOL:
            np_type = bool
        else:
            return None
        return helper.make_node('Constant', inputs=[], outputs=[name],
                                value=helper.make_tensor(name='const_tensor',
                                                         data_type=tensor_type,
                                                         dims=value.shape,
                                                         vals=value.flatten().astype(np_type)))

    @staticmethod
    def create_body_graph(input_nodes, output_nodes, input_names, output_names, input_shape,
                          graph_name):
        # input_nodes - list of input nodes with structure {counter, condition, <other inputs>}
        # output_nodes - list of output nodes with structure {condition, <back edges>, <external outputs>}.
        # In this function I assume that every <other input> have <back edge> and <external output>
        # input_shape - shape of all inputs from <other inputs>
        from onnx import helper
        from onnx import TensorProto

        assert len(input_nodes) > 2
        assert len(output_nodes) == (len(input_nodes) - 2) * 2 + 1
        assert len(input_nodes) == len(input_names)
        assert len(output_nodes) == len(output_names)
        other_inputs_count = len(input_nodes) - 2
        one_value = np.ones(input_shape, dtype=float)

        one = TestLoop.create_const('one_' + graph_name, TensorProto.FLOAT, one_value)
        one_int = TestLoop.create_const('one_int_' + graph_name, TensorProto.INT64, np.ones([1]))

        # add one to all inputs except counter and condition
        add_one_nodes = []
        for i in range(2, len(input_names)):
            add_one_nodes.append(
                helper.make_node('Add', inputs=[input_names[i], 'one_' + graph_name],
                                 outputs=[output_names[other_inputs_count + i - 1]]))

        # add 1 to counter
        add_one_to_m_node = helper.make_node(
            'Add',
            inputs=[input_names[0], 'one_int_' + graph_name],
            outputs=['counter_plus_1_' + graph_name]
        )

        # map inputs to outputs - back edges
        identity_nodes = []
        for i in range(1, len(input_nodes)):
            identity_nodes.append(helper.make_node('Identity',
                                                   inputs=[input_names[i]],
                                                   outputs=[output_names[i - 1]]))

        body_nodes = [one, one_int]
        body_nodes.extend(add_one_nodes)
        body_nodes.append(add_one_to_m_node)
        body_nodes.extend(identity_nodes)
        body_graph = helper.make_graph(
            body_nodes,
            graph_name,
            input_nodes,
            output_nodes
        )

        return body_graph

    def create_loop(self):
        """
            ONNX net

            Input->Loop->Output   =>   Only accuracy check

        """
        from onnx import helper
        from onnx import TensorProto

        #   Create ONNX model
        #   Input ---> Loop ---> Identity ---> Result
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
        cond_value = np.array([True], bool)

        M_1 = self.create_const('M_1', TensorProto.INT64, m_1_value)
        cond = self.create_const('cond', TensorProto.BOOL, cond_value)

        body_graph_1 = self.create_body_graph([m_1, cond_int_1, in_1_int],
                                              [cond_out_1, in_1_int_out, out_1],
                                              ['m_1', 'cond_int_1', 'in_1_int'],
                                              ['cond_out_1', 'in_1_int_out', 'OUT_1'],
                                              input_shape, 'body_graph_1')

        node_loop_1 = helper.make_node(
            'Loop',
            inputs=['M_1', 'cond', 'IN_1'],
            outputs=['cond_out_1', 'OUT_1'],
            body=body_graph_1
        )

        res_node = helper.make_node(
            'Identity',
            inputs=['OUT_1'],
            outputs=['res'],
        )

        graph_def = helper.make_graph(
            [M_1, cond, node_loop_1, res_node],
            'graph',
            [in_1],
            [res]
        )

        onnx_net = onnx_make_model(graph_def, producer_name='test_loop_model')
        # We do not create reference graph, as it's too complicated to construct it
        # So we return None to skip IR comparision
        return onnx_net, None

    def create_loop_in_loop(self):
        """
            ONNX net

            Input->Loop(Loop)->Output   =>   Only accuracy check

        """
        from onnx import helper
        from onnx import TensorProto

        #   Create ONNX model
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
        cond_value = np.array([True], bool)
        one_value = np.ones(input_shape, dtype=float)

        M_1 = self.create_const('M_1', TensorProto.INT64, m_1_value)
        M_2 = self.create_const('M_2', TensorProto.INT64, m_2_value)
        cond = self.create_const('cond', TensorProto.BOOL, cond_value)
        one = self.create_const('one', TensorProto.FLOAT, one_value)
        one_int = self.create_const('one_int', TensorProto.INT64, one_value)

        # create body of external loop
        add_one_node = helper.make_node(
            'Add',
            inputs=['in_1_int', 'one'],
            outputs=['in_1_loop_1']
        )

        add_one_to_m_node = helper.make_node(
            'Add',
            inputs=['m_1', 'one_int'],
            outputs=['m_1_loop_1']
        )

        cond_2 = self.create_const('cond_2', TensorProto.BOOL, cond_value)

        # create body for internal loop
        body_graph_2 = self.create_body_graph([m_2, cond_int_2, in_2_int],
                                              [cond_out_2, in_2_int_out, out_2],
                                              ['m_2', 'cond_int_2', 'in_2_int'],
                                              ['cond_out_2', 'in_2_int_out', 'OUT_2'], input_shape,
                                              'body_graph_2')
        node_loop_2 = helper.make_node(
            'Loop',
            inputs=['M_2', 'cond_2', 'IN_2'],
            outputs=['cond_out_2', 'OUT_2'],
            body=body_graph_2
        )
        # internal loop created

        out_1_node = helper.make_node(
            'Identity',
            inputs=['OUT_2'],
            outputs=['OUT_1'],
        )

        cond_1_node = helper.make_node(
            'Identity',
            inputs=['cond_int_1'],
            outputs=['cond_out_1'],
        )

        in_1_int_node = helper.make_node(
            'Identity',
            inputs=['in_1_int'],
            outputs=['in_1_int_out'],
        )

        body_graph_1 = helper.make_graph(
            [one, add_one_node, one_int, add_one_to_m_node, M_2, cond_2, node_loop_2, out_1_node,
             cond_1_node,
             in_1_int_node],
            'body_graph_1',
            [m_1, cond_int_1, in_1_int],
            [cond_out_1, in_1_int_out, out_1],
        )

        node_loop_1 = helper.make_node(
            'Loop',
            inputs=['M_1', 'cond', 'IN_1'],
            outputs=['cond_out_1', 'OUT_1'],
            body=body_graph_1
        )
        # external loop created

        res_node = helper.make_node(
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

        onnx_net = onnx_make_model(graph_def, producer_name='test_loop_in_loop_model')
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
        if ie_device == 'GPU':
            pytest.xfail("Program doesn't contain primitive: constant:res/10/M_2 that is input to: loop")
        self._test(*self.create_loop_in_loop(), ie_device, precision, ir_version, temp_dir=temp_dir,
                   infer_timeout=150)
