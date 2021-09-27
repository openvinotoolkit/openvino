# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.ConvertLike import ConvertLike
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.ops.const import Const
from mo.ops.pad import Pad


class AttributedPadToPad(FrontReplacementPattern):
    """
    This transformation converts AttributedPad operation (begin/end paddings are specified as attribute) to Pad
    operation (Inference Engine semantic).
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for attr_pad in graph.get_op_nodes(op='AttributedPad'):
            # save the original node name to use it in the new Pad op instance
            original_name = attr_pad.soft_get('name', attr_pad.id)

            new_pad = Pad(graph, {'mode': attr_pad.soft_get('mode', None), }).create_node()
            rename_nodes([(attr_pad, original_name + '/to_be_removed'), (new_pad, original_name)])

            attr_pad.in_port(0).get_connection().set_destination(new_pad.in_port(0))
            new_pad.in_port(1).connect(Const(graph, {'value': attr_pad.pads[:, 0]}).create_node().out_port(0))
            new_pad.in_port(2).connect(Const(graph, {'value': attr_pad.pads[:, 1]}).create_node().out_port(0))
            if attr_pad.soft_get('mode') == 'constant':
                # create Constant node of proper data type (equal to the data type of the Pad first input)
                convert_pad_value = create_op_with_const_inputs(graph, ConvertLike, {0: attr_pad.fill_value},
                                                                {'name': original_name + '/pad_value_convert'})
                convert_pad_value.in_port(1).connect(new_pad.in_port(0).get_source())
                new_pad.in_port(3).connect(convert_pad_value.out_port(0))

            attr_pad.out_port(0).get_connection().set_source(new_pad.out_port(0))
            graph.remove_node(attr_pad.id)
