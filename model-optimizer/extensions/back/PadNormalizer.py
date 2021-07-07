# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.ConvertLike import ConvertLike
from mo.back.replacement import BackReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph


class PadNormalizer(BackReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.back.ReverseInputChannels import ApplyReverseChannels
        return [ApplyReverseChannels]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='Pad'):
            name = node.soft_get('name', node.id)
            if node.soft_get('mode') == 'constant' and not node.is_in_port_connected(3):
                # create Constant node of proper data type (equal to the data type of the Pad first input)
                convert_pad_value = create_op_with_const_inputs(graph, ConvertLike, {0: 0.0},
                                                                {'name': name + '/pad_value_convert'})
                convert_pad_value.in_port(1).connect(node.in_port(0).get_source())
                node.add_input_port(3, skip_if_exist=True)
                node.in_port(3).connect(convert_pad_value.out_port(0))
