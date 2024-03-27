# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.middle.LSTMRNNSequenceToTensorIterator import LSTMToTensorIterator
from openvino.tools.mo.middle.ONNXRNNSequenceNormalize import ONNXRNNSequenceNormalize
from openvino.tools.mo.middle.SwapAxesMiddleReplacer import SwapAxisMiddleReplacer
from openvino.tools.mo.middle.TensorIteratorMerge import TensorIteratorMerge
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import dict_includes, Graph
from openvino.tools.mo.middle.passes.eliminate import remove_op_node_with_data_node
from openvino.tools.mo.middle.pattern_match import find_isomorphisms
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class TransposeTensorIteratorLSTM(MiddleReplacementPattern):
    """ Fuses Transpose(1,0,2) --> TI --> Transpose(1,0,2) pattern to a single TI with changed axis.

        WARNING This transformation is limited to support of very special case of TI but
        code doesn't check all the cases.
    """

    enabled = True

    def run_after(self):
        return [TensorIteratorMerge, ONNXRNNSequenceNormalize, LSTMToTensorIterator, SwapAxisMiddleReplacer]

    def run_before(self):
        return []

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict(kind='data')),
                ('direct_permute', dict(kind='op', op='Transpose')),
                ('input_permuted', dict(kind='data')),
                ('init_hidden', dict(kind='data')),
                ('init_cell', dict(kind='data')),
                ('ti', dict(kind='op', op='TensorIterator')),

                ('output_permuted', dict(kind='data')),
                ('inverse_permute', dict(op='Transpose')),
                ('output', dict(kind='data')),
            ],
            edges=[
                ('input', 'direct_permute'),
                ('direct_permute', 'input_permuted'),

                ('input_permuted', 'ti', {'in': 0}),  # affected by permute
                ('init_hidden', 'ti', {'in': 1}),
                ('init_cell', 'ti', {'in': 2}),
                ('ti', 'output_permuted', {'out': 0}),  # affected by permute

                ('output_permuted', 'inverse_permute'),
                ('inverse_permute', 'output'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):

        # This transformation works if and only if a body of TI
        # matches the following topology (Squeeze -> LSTMCell -> Unsqueeze)
        nodes = [
            ('squeeze_dim', dict(kind='op', op='Const')),
            ('squeeze_dim_data', dict(kind='data')),

            ('unsqueeze_dim', dict(kind='op', op='Const')),
            ('unsqueeze_dim_data', dict(kind='data')),

            ('input_unsqueezed', dict(kind='data')),
            ('squeeze', dict(kind='op', op='Squeeze')),
            ('input_squeezed', dict(kind='data')),
            ('input_hidden', dict(kind='data')),
            ('input_cell', dict(kind='data')),
            ('weights', dict(kind='data')),
            ('biases', dict(kind='data')),

            ('lstm', dict(kind='op', op='LSTMCell')),

            ('output_hidden', dict(kind='data')),
            ('output_cell', dict(kind='data')),
            ('unsqueeze', dict(kind='op', op='Unsqueeze')),
            ('output_unsqueezed', dict(kind='data')),

            ('const_w', dict(kind='op', op='Const')),
            ('const_b', dict(kind='op', op='Const')),

            ('op_output', dict(kind='op', op='Result')),
            ('op_output_1', dict(kind='op', op='Result')),
            ('op_output_2', dict(kind='op', op='Result')),

            ('input_unsqueezed_i', dict(kind='op', op='Parameter')),
            ('input_hidden_i', dict(kind='op', op='Parameter')),
            ('input_cell_i', dict(kind='op', op='Parameter')),
        ]
        edges = [
            ('input_unsqueezed', 'squeeze', {'in': 0}),
            ('squeeze', 'input_squeezed'),

            ('squeeze_dim', 'squeeze_dim_data'),
            ('squeeze_dim_data', 'squeeze', {'in': 1}),

            ('input_squeezed', 'lstm', {'in': 0}),
            ('input_hidden', 'lstm', {'in': 1}),
            ('input_cell', 'lstm', {'in': 2}),
            ('weights', 'lstm', {'in': 3}),
            ('biases', 'lstm', {'in': 4}),

            ('const_w', 'weights'),
            ('const_b', 'biases'),

            ('lstm', 'output_hidden', {'out': 0}),
            ('lstm', 'output_cell', {'out': 1}),

            ('output_hidden', 'unsqueeze'),
            ('unsqueeze', 'output_unsqueezed'),

            ('unsqueeze_dim', 'unsqueeze_dim_data'),
            ('unsqueeze_dim_data', 'unsqueeze', {'in': 1}),

            ('output_unsqueezed', 'op_output'),
            ('output_hidden', 'op_output_1'),
            ('output_cell', 'op_output_2'),

            ('input_unsqueezed_i', 'input_unsqueezed'),
            ('input_hidden_i', 'input_hidden'),
            ('input_cell_i', 'input_cell'),
        ]
        ti = match['ti']
        isomorphisms = find_isomorphisms(ti.body, nodes, edges)
        if len(list(isomorphisms)) != 1:
            return
        isomorphism = isomorphisms[0]

        direct_permute = match['direct_permute']
        inverse_permute = match['inverse_permute']

        permute_order = [1, 0, 2]

        # Check both perumute orders exactly match expected one - [1, 0, 2]
        direct_order = direct_permute.in_port(1).data.get_value()
        if direct_order is None or not np.array_equal(direct_order, permute_order):
            return
        inverse_order = inverse_permute.in_port(1).data.get_value()
        if inverse_order is None or not np.array_equal(inverse_order, permute_order):
            return

        # Check non-ShapeOf output out of direct Transpose is exactly one
        direct_permute_dsts = direct_permute.out_port(0).get_destinations()
        if len([dst for dst in direct_permute_dsts if dst.node.soft_get('type') != 'ShapeOf']) != 1:
            return
        for shape_of_dst in [dst for dst in direct_permute_dsts if dst.node.soft_get('type') == 'ShapeOf']:
            name = shape_of_dst.node.soft_get('name', shape_of_dst.node.id) + '/FusedToTITranspose'
            gather = create_op_with_const_inputs(graph, op=Gather, op_attrs={'name': name},
                                                 port_value_dict={1: int64_array(permute_order), 2: int64_array(0)})
            shape_of_dst.node.out_port(0).get_connection().insert_node(gather)

        def find_ports(port_map: list, attrs: dict):
            """ Find all ports in a given port map with specified attributes """
            result = []
            for i, port in enumerate(port_map):
                if dict_includes(port, attrs):
                    result.append(i)
            return result

        # Check TI has only single partitioned input/output port; all partitioned ports have defined axis
        data_input_port = find_ports(ti.input_port_map, {'axis': lambda attr: attr in [0, 1]})
        data_output_port = find_ports(ti.output_port_map, {'axis': lambda attr: attr in [0, 1]})
        assert len(data_input_port) == 1
        assert len(data_output_port) == 1
        data_input_port = data_input_port[0]
        data_output_port = data_output_port[0]
        # Verify that they are really connected to Transpose layers (guaranteed by port numbers of TI, see the pattern)
        assert ti.in_edge(0)['external_port_id'] == ti.input_port_map[data_input_port]['external_port_id']
        assert ti.out_edge(0)['external_port_id'] == ti.output_port_map[data_output_port]['external_port_id']

        # Verify that the TI body have required Reshapes connected to the found ports
        squeeze = isomorphism['squeeze']
        unsqueeze = isomorphism['unsqueeze']

        assert len(squeeze.in_node().shape) == 3
        assert len(squeeze.out_node().shape) == 2
        assert len(unsqueeze.in_node().shape) == 2
        assert len(unsqueeze.out_node().shape) == 3

        # Remove permutes
        remove_op_node_with_data_node(graph, direct_permute)
        remove_op_node_with_data_node(graph, inverse_permute)
        match['output'].shape = match['output'].shape[permute_order]

        # swap 0/1 axis for partitioned ports
        ti.input_port_map[data_input_port]['axis'] = 1 - ti.input_port_map[data_input_port]['axis']
        ti.output_port_map[data_output_port]['axis'] = 1 - ti.output_port_map[data_output_port]['axis']

        isomorphism['input_unsqueezed_i'].shape = isomorphism['input_unsqueezed_i'].shape[[1, 0, 2]]
        isomorphism['input_unsqueezed_i'].infer(isomorphism['input_unsqueezed_i'])
        isomorphism['squeeze_dim'].value = ti.input_port_map[data_input_port]['axis']
        isomorphism['squeeze_dim'].infer(isomorphism['squeeze_dim'])
        isomorphism['squeeze']['need_shape_inference'] = True

        isomorphism['unsqueeze_dim'].value = ti.output_port_map[data_output_port]['axis']
        isomorphism['unsqueeze_dim'].infer(isomorphism['unsqueeze_dim'])
        isomorphism['unsqueeze'].infer(isomorphism['unsqueeze'])
